"""Micro-hole filling post-processing for thin ribbon segmentation.

Targets 1-2 voxel holes/pits that appear inside predicted ribbons.
These micro-holes create spurious background connected components which
dramatically hurt topological (Betti) scores.

Key insight: The Kaggle topo metric inverts FG/BG before computing Betti
numbers, so micro-holes in the foreground become extra background connected
components (dim 0), tanking the score.

Diagnostic findings (2026-02-09):
- Samples have 0-78 small BG components (≤100 voxels each)
- 2D binary_fill_holes per slice eliminates ALL small BG components
- remove_small_holes (3D) also eliminates them efficiently
- 3D binary_closing is COUNTERPRODUCTIVE: removes voxels and can create
  new small BG components due to the erosion step

Additional finding: The DOMINANT topo issue is too many FG connected
components (predictions have +2 to +15 more components than GT). This is
caused by the model predicting spurious extra ribbons. We need to either:
a) Remove excess small FG components (increase min_size filter)
b) Fill micro-holes to fix background topology
Both approaches combined should improve topo.

Created: 2026-02-09
"""

from __future__ import annotations

import logging

import cc3d
import numpy as np
from scipy.ndimage import (
    binary_fill_holes,
)
from scipy.ndimage import (
    label as scipy_label,
)
from skimage.morphology import remove_small_holes, remove_small_objects

logger = logging.getLogger(__name__)


def fill_microholes_2d(mask: np.ndarray) -> np.ndarray:
    """Fill holes in each z-slice independently.

    This catches small 2D holes (pits) within each slice of the ribbon.
    Fast: operates slice by slice with scipy binary_fill_holes.
    """
    result = mask.copy()
    for z in range(mask.shape[0]):
        result[z] = binary_fill_holes(mask[z]).astype(mask.dtype)
    return result


def fill_microholes_3d_small(
    mask: np.ndarray,
    max_hole_size: int = 64,
) -> np.ndarray:
    """Remove small 3D holes (enclosed background regions).

    Uses skimage remove_small_holes to fill background regions
    smaller than max_hole_size voxels. This directly targets
    the micro-holes that create spurious background components.

    Args:
        mask: Binary mask (uint8)
        max_hole_size: Maximum hole size in voxels to fill.
            64 = fills holes up to ~4x4x4 voxels.
            27 = fills holes up to ~3x3x3 voxels.
            8 = fills holes up to ~2x2x2 voxels.

    Returns:
        Mask with small holes filled

    """
    return remove_small_holes(
        mask.astype(bool),
        area_threshold=max_hole_size,
    ).astype(np.uint8)


def adaptive_border_aware_holefill(
    pr: np.ndarray,
    adaptive_method: str = "learned",
    min_size: int = 500,
    border_standoff: int = 15,
    require_both_endpoints: bool = False,
    fill_2d: bool = True,
    fill_3d_holes: bool = True,
    max_hole_size: int = 100,
    **kwargs,
) -> np.ndarray:
    """Adaptive border-aware + micro-hole filling.

    Combines the proven adaptive_border_aware approach with targeted
    micro-hole filling to improve topological scores.

    Pipeline:
    1. Adaptive thresholding (learned)
    2. Border endpoint filtering
    3. 2D per-slice hole filling (removes 2D pits)
    4. 3D small hole removal (removes tiny enclosed BG regions)
    5. Re-apply small component removal

    Diagnostic results show this eliminates ALL small BG components
    (the micro-holes) without risking merging separate ribbons.

    Args:
        pr: Probability map (float32, 0-1)
        adaptive_method: Threshold prediction method
        min_size: Minimum component size in voxels
        border_standoff: Border distance for endpoint filtering
        require_both_endpoints: Whether both endpoints must be near border
        fill_2d: Whether to fill 2D holes per slice
        fill_3d_holes: Whether to remove small 3D holes
        max_hole_size: Maximum 3D hole size to fill (voxels)

    Returns:
        Binary mask (uint8)

    """
    # Lazy imports to avoid circular deps
    from analyze_prediction_probabilities import adaptive_threshold_postprocess
    from post_processing_probs import filter_by_border_endpoints

    # Step 1: Adaptive thresholding
    mask = adaptive_threshold_postprocess(pr, method=adaptive_method, min_size=min_size)

    # Step 2: Border endpoint filtering
    min_endpoints = 2 if require_both_endpoints else 1
    mask = filter_by_border_endpoints(
        mask,
        border_standoff=border_standoff,
        require_start_near_border=require_both_endpoints,
        require_end_near_border=require_both_endpoints,
        min_endpoint_near_border=min_endpoints,
    )

    # Step 3: Micro-hole filling
    if fill_2d:
        mask = fill_microholes_2d(mask)
    if fill_3d_holes:
        mask = fill_microholes_3d_small(mask, max_hole_size=max_hole_size)

    # Step 4: Re-apply small component removal (hole filling may reconnect
    # tiny fragments that were previously separate)
    mask = remove_small_objects(
        mask.astype(bool),
        min_size=min_size,
        connectivity=3,
    ).astype(np.uint8)

    return mask


def adaptive_border_aware_holefill_v2(
    pr: np.ndarray,
    adaptive_method: str = "learned",
    min_size: int = 500,
    border_standoff: int = 15,
    require_both_endpoints: bool = False,
    max_hole_size: int = 100,
    aggressive_min_size: int = 3000,
    min_z_span: int = 10,
    **kwargs,
) -> np.ndarray:
    """V2: Hole filling + aggressive small component removal.

    Key insight from diagnostic: the dominant topo issue is too many
    FG connected components (predictions have +2 to +15 extra vs GT).
    This version aggressively removes small/flat components AND fills holes.

    Pipeline:
    1. Adaptive thresholding
    2. Border endpoint filtering
    3. Fill micro-holes (2D + 3D)
    4. Aggressive component filtering:
       - Remove components < aggressive_min_size voxels
       - Remove components with z-span < min_z_span slices
    """
    from analyze_prediction_probabilities import adaptive_threshold_postprocess
    from post_processing_probs import filter_by_border_endpoints

    # Step 1: Adaptive thresholding
    mask = adaptive_threshold_postprocess(pr, method=adaptive_method, min_size=min_size)

    # Step 2: Border endpoint filtering
    min_endpoints = 2 if require_both_endpoints else 1
    mask = filter_by_border_endpoints(
        mask,
        border_standoff=border_standoff,
        require_start_near_border=require_both_endpoints,
        require_end_near_border=require_both_endpoints,
        min_endpoint_near_border=min_endpoints,
    )

    # Step 3: Fill micro-holes
    mask = fill_microholes_2d(mask)
    mask = fill_microholes_3d_small(mask, max_hole_size=max_hole_size)

    # Step 4: Aggressive component filtering
    # Use cc3d for fast component analysis
    labels = cc3d.connected_components(mask.astype(np.uint8), connectivity=26)
    stats = cc3d.statistics(labels)
    result = np.zeros_like(mask)

    for label_id in range(1, stats["voxel_counts"].shape[0]):
        voxel_count = stats["voxel_counts"][label_id]

        # Size filter
        if voxel_count < aggressive_min_size:
            continue

        # Z-span filter
        component = labels == label_id
        z_indices = np.where(component.any(axis=(1, 2)))[0]
        if len(z_indices) < min_z_span:
            continue

        result[component] = 1

    return result.astype(np.uint8)


def adaptive_border_aware_holefill_v3(
    pr: np.ndarray,
    adaptive_method: str = "learned",
    min_size: int = 500,
    border_standoff: int = 15,
    require_both_endpoints: bool = False,
    max_hole_size: int = 200,
    **kwargs,
) -> np.ndarray:
    """V3: Aggressive hole filling (2D fill + larger 3D threshold).

    More aggressive hole filling:
    - 2D per-slice hole filling
    - Larger max_hole_size for 3D (200 voxels)

    Best for samples with many holes of varying sizes.
    """
    from analyze_prediction_probabilities import adaptive_threshold_postprocess
    from post_processing_probs import filter_by_border_endpoints

    # Step 1: Adaptive thresholding
    mask = adaptive_threshold_postprocess(pr, method=adaptive_method, min_size=min_size)

    # Step 2: Border endpoint filtering
    min_endpoints = 2 if require_both_endpoints else 1
    mask = filter_by_border_endpoints(
        mask,
        border_standoff=border_standoff,
        require_start_near_border=require_both_endpoints,
        require_end_near_border=require_both_endpoints,
        min_endpoint_near_border=min_endpoints,
    )

    # Step 3: 2D hole fill
    mask = fill_microholes_2d(mask)

    # Step 4: Large 3D hole removal
    mask = fill_microholes_3d_small(mask, max_hole_size=max_hole_size)

    # Step 5: Clean up
    mask = remove_small_objects(
        mask.astype(bool),
        min_size=min_size,
        connectivity=3,
    ).astype(np.uint8)

    return mask
