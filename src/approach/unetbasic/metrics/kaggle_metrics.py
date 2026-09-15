"""Kaggle competition metrics for Vesuvius Challenge.

This module provides utilities to compute the official competition metric
using the topometrics package during training evaluation.
"""

import gc
import multiprocessing as mp
import signal
import warnings
from multiprocessing import cpu_count
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from monai import transforms as MT
from scipy import ndimage

# Import topometrics - already installed in the environment
from topometrics.leaderboard import compute_leaderboard_score

# Use spawn context for clean memory state in worker processes
# This prevents memory leaks from parent process state
_MP_CONTEXT = mp.get_context("spawn")

# Default metric weights
DEFAULT_TOPO_WEIGHT = 0.30
DEFAULT_SURFACE_DICE_WEIGHT = 0.35
DEFAULT_VOI_WEIGHT = 0.35
DEFAULT_SURFACE_TOLERANCE = 2.0
DEFAULT_VOI_CONNECTIVITY = 26
DEFAULT_VOI_TRANSFORM = "one_over_one_plus"
DEFAULT_VOI_ALPHA = 0.3
BINARY_THRESHOLD = 0.5


def _worker_init() -> None:
    """Initialize worker process with proper signal handling."""
    # Ignore SIGINT in workers - let parent handle it
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def apply_morphological_postprocessing(
    predictions: np.ndarray,
    remove_small_objects_threshold: int = 50,
    remove_small_holes_threshold: int = 50,
    apply_closing: bool = True,
    closing_iterations: int = 1,
) -> np.ndarray:
    """Apply morphological post-processing to improve topology.

    This can significantly improve topological scores by:
    1. Removing small disconnected objects (noise)
    2. Filling small holes (improves connectivity)
    3. Closing operations (smooths boundaries, connects nearby regions)

    Args:
        predictions: Binary predictions (0/1)
        remove_small_objects_threshold: Minimum object size in voxels (0 to disable)
        remove_small_holes_threshold: Maximum hole size to fill (0 to disable)
        apply_closing: Whether to apply morphological closing
        closing_iterations: Number of closing iterations

    Returns:
        Post-processed binary predictions

    """
    from scipy.ndimage import binary_closing, binary_fill_holes

    # Work on a copy
    processed = predictions.copy()

    # 1. Remove small objects (noise reduction)
    if remove_small_objects_threshold > 0:
        labeled, num_features = ndimage.label(processed)
        if num_features > 0:
            sizes = ndimage.sum(processed, labeled, range(num_features + 1))
            mask_sizes = sizes >= remove_small_objects_threshold
            # Remove small objects
            processed = mask_sizes[labeled]

    # 2. Fill small holes (improves connectivity)
    if remove_small_holes_threshold > 0:
        # Fill holes in each connected component separately
        labeled, num_features = ndimage.label(processed)
        for i in range(1, num_features + 1):
            component = labeled == i
            # Fill holes in this component
            filled = binary_fill_holes(component)
            # Only keep small holes that were filled
            holes = filled & ~component
            holes_labeled, n_holes = ndimage.label(holes)
            if n_holes > 0:
                hole_sizes = ndimage.sum(holes, holes_labeled, range(n_holes + 1))
                small_holes_mask = hole_sizes <= remove_small_holes_threshold
                small_holes = small_holes_mask[holes_labeled]
                processed = processed | (holes & small_holes)

    # 3. Morphological closing (smooth boundaries, connect nearby regions)
    if apply_closing and closing_iterations > 0:
        # Use a 3x3x3 structuring element
        structure = ndimage.generate_binary_structure(3, connectivity=1)
        processed = binary_closing(processed, structure=structure, iterations=closing_iterations)

    return processed.astype(predictions.dtype)


def apply_gaussian_smoothing(
    predictions: np.ndarray,
    sigma: float = 0.5,
) -> np.ndarray:
    """Apply Gaussian smoothing before thresholding.

    This can help by:
    - Reducing noise in probability maps
    - Creating smoother, more connected regions
    - Better matching the spatial structure of ground truth

    Args:
        predictions: Probability predictions [0, 1]
        sigma: Standard deviation for Gaussian kernel

    Returns:
        Smoothed predictions [0, 1]

    """
    smoothed = ndimage.gaussian_filter(predictions, sigma=sigma)
    # Ensure values stay in [0, 1] range
    return np.clip(smoothed, 0.0, 1.0)


def apply_adaptive_histogram_equalization(
    predictions: np.ndarray,
    clip_limit: float = 0.03,
) -> np.ndarray:
    """Apply CLAHE to enhance contrast in prediction probabilities.

    This can help by:
    - Enhancing local contrast in probability maps
    - Making weak predictions more decisive
    - Better separating foreground from background

    Args:
        predictions: Probability predictions [0, 1]
        clip_limit: Clipping limit for contrast enhancement

    Returns:
        Enhanced predictions [0, 1]

    """
    from skimage import exposure

    # CLAHE works on uint8, so convert
    pred_uint8 = (predictions * 255).astype(np.uint8)
    enhanced = exposure.equalize_adapthist(pred_uint8, clip_limit=clip_limit)
    return enhanced.astype(np.float32)


def apply_connected_component_refinement(
    predictions: np.ndarray,
    min_size_percentile: float = 5.0,
) -> np.ndarray:
    """Keep only significant connected components based on size distribution.

    Instead of fixed threshold, this uses statistics of component sizes.

    Args:
        predictions: Binary predictions (0/1)
        min_size_percentile: Keep components larger than this percentile

    Returns:
        Refined binary predictions

    """
    labeled, num_features = ndimage.label(predictions)
    if num_features == 0:
        return predictions

    # Get sizes of all components
    sizes = np.array([np.sum(labeled == i) for i in range(1, num_features + 1)])

    if len(sizes) == 0:
        return predictions

    # Keep components larger than percentile threshold
    size_threshold = np.percentile(sizes, min_size_percentile)

    # Create mask for significant components
    refined = np.zeros_like(predictions)
    for i in range(1, num_features + 1):
        if sizes[i - 1] >= size_threshold:
            refined[labeled == i] = 1

    return refined.astype(predictions.dtype)


#################################################################################
# More complex functions below - do not edit unless necessary
#################################################################################
def apply_directional_hole_filling(
    predictions: np.ndarray,
    max_hole_size: int = 100,
    axis: int = 0,
) -> np.ndarray:
    """Fill small holes in sheets using slice-by-slice 2D closing.

    This exploits the fact that sheets are planar structures that span
    the H×W plane. By processing slice-by-slice along Z (axis 0), we can
    fill small holes without accidentally bridging separate sheets.

    Args:
        predictions: Binary predictions (0/1) with shape (D, H, W)
        max_hole_size: Maximum hole size in pixels to fill (2D area per slice)
        axis: Axis along which to process slices (0=Z, 1=H, 2=W)

    Returns:
        Post-processed binary predictions with holes filled

    """
    from scipy.ndimage import binary_closing, binary_fill_holes
    from scipy.ndimage import label as label_2d

    processed = predictions.copy()

    # Create a 2D structuring element for in-plane closing
    structure_2d = ndimage.generate_binary_structure(2, connectivity=1)

    n_slices = processed.shape[axis]

    for i in range(n_slices):
        # Extract 2D slice
        if axis == 0:
            slice_2d = processed[i, :, :]
        elif axis == 1:
            slice_2d = processed[:, i, :]
        else:
            slice_2d = processed[:, :, i]

        # Apply 2D closing to connect nearby regions within the slice
        closed = binary_closing(slice_2d, structure=structure_2d, iterations=2)

        # Fill small holes in the closed result
        # First find all holes by comparing with filled version
        filled = binary_fill_holes(closed)
        holes = filled & ~closed

        if holes.any():
            # Label holes and filter by size
            labeled_holes, n_holes = label_2d(holes)
            if n_holes > 0:
                hole_sizes = ndimage.sum(holes, labeled_holes, range(1, n_holes + 1))
                # Only fill small holes
                for hole_idx, hole_size in enumerate(hole_sizes, start=1):
                    if hole_size <= max_hole_size:
                        closed = closed | (labeled_holes == hole_idx)

        # Write back to volume
        if axis == 0:
            processed[i, :, :] = closed
        elif axis == 1:
            processed[:, i, :] = closed
        else:
            processed[:, :, i] = closed

    return processed.astype(predictions.dtype)


def apply_sheet_thinning(
    predictions: np.ndarray,
    target_thickness: int = 6,
    min_thickness: int = 4,
) -> np.ndarray:
    """Reduce over-thick sheet predictions to target thickness.

    Uses distance transform to identify the medial surface of thick regions,
    then dilates to achieve target thickness. This preserves connectivity
    while reducing bloated predictions.

    Args:
        predictions: Binary predictions (0/1)
        target_thickness: Target sheet thickness in voxels (diameter, not radius)
        min_thickness: Minimum thickness to preserve (don't thin below this)

    Returns:
        Thinned binary predictions

    """
    from scipy.ndimage import binary_dilation, binary_erosion, distance_transform_edt

    if predictions.sum() == 0:
        return predictions

    # Compute distance transform - distance from each foreground voxel to nearest background
    dist = distance_transform_edt(predictions)

    # Find the maximum thickness (diameter = 2 * max_distance)
    max_dist = dist.max()
    current_max_thickness = int(2 * max_dist)

    # If already thin enough, return as-is
    if current_max_thickness <= target_thickness:
        return predictions

    # Calculate how much to erode
    # We want final thickness = target_thickness
    # Current thickness at thickest point = 2 * max_dist
    # Erosion removes 1 voxel from each side, so erosion of N reduces thickness by 2N
    target_radius = target_thickness / 2.0
    erosion_amount = max(0, int(max_dist - target_radius))

    if erosion_amount == 0:
        return predictions

    # Erode to thin down
    structure = ndimage.generate_binary_structure(3, connectivity=1)
    eroded = binary_erosion(predictions, structure=structure, iterations=erosion_amount)

    # Check if erosion disconnected components - if so, use gentler approach
    original_labels, original_count = ndimage.label(predictions)
    eroded_labels, eroded_count = ndimage.label(eroded)

    # If we lost connectivity, back off erosion
    if eroded_count > original_count * 1.5 or eroded.sum() < predictions.sum() * 0.3:
        # Too aggressive - try half the erosion
        erosion_amount = max(1, erosion_amount // 2)
        eroded = binary_erosion(predictions, structure=structure, iterations=erosion_amount)

        eroded_labels, eroded_count = ndimage.label(eroded)
        # Still too aggressive? Just do minimal erosion
        if eroded_count > original_count * 1.5 or eroded.sum() < predictions.sum() * 0.3:
            eroded = binary_erosion(predictions, structure=structure, iterations=1)

    # Ensure minimum thickness by dilating if needed
    if eroded.sum() > 0:
        new_dist = distance_transform_edt(eroded)
        if new_dist.max() * 2 < min_thickness:
            # Need to dilate a bit to maintain minimum thickness
            dilation_needed = max(1, int((min_thickness / 2) - new_dist.max()))
            eroded = binary_dilation(eroded, structure=structure, iterations=dilation_needed)

    return eroded.astype(predictions.dtype)


def apply_z_continuity_enforcement(
    predictions: np.ndarray,
    window_size: int = 5,
    threshold: float = 0.6,
    fill_gaps: bool = True,
    remove_noise: bool = True,
) -> np.ndarray:
    """Enforce continuity along Z-axis using sliding window voting.

    Sheets should be continuous along the scroll length (Z-axis). This function:
    1. Fills gaps where a voxel is background but neighbors along Z are foreground
    2. Removes noise where a voxel is foreground but neighbors along Z are background

    Args:
        predictions: Binary predictions (0/1) with shape (D, H, W)
        window_size: Size of sliding window along Z (must be odd)
        threshold: Fraction of window that must agree (0.5-1.0)
        fill_gaps: If True, fill background voxels with strong Z-neighbors
        remove_noise: If True, remove foreground voxels with weak Z-neighbors

    Returns:
        Post-processed predictions with Z-continuity enforced

    """
    if window_size % 2 == 0:
        window_size += 1  # Ensure odd

    half_window = window_size // 2
    D, H, W = predictions.shape

    processed = predictions.copy()

    # Compute Z-axis vote for each voxel
    # This counts how many voxels in the Z-window are foreground
    z_votes = np.zeros_like(predictions, dtype=np.float32)

    for dz in range(-half_window, half_window + 1):
        # Shift predictions along Z and accumulate
        if dz < 0:
            z_votes[-dz:, :, :] += predictions[:dz, :, :]
        elif dz > 0:
            z_votes[:-dz, :, :] += predictions[dz:, :, :]
        else:
            z_votes += predictions

    # Normalize to get fraction
    z_votes /= window_size

    # Handle edge cases where window extends beyond volume
    for z in range(half_window):
        actual_window = half_window + z + 1
        z_votes[z, :, :] *= window_size / actual_window
    for z in range(D - half_window, D):
        actual_window = half_window + (D - z)
        z_votes[z, :, :] *= window_size / actual_window

    # Fill gaps: background voxels with strong Z-support become foreground
    if fill_gaps:
        fill_mask = (predictions == 0) & (z_votes >= threshold)
        processed[fill_mask] = 1

    # Remove noise: foreground voxels with weak Z-support become background
    if remove_noise:
        noise_threshold = 1.0 - threshold  # If 60% must agree, remove if <40% agree
        noise_mask = (predictions == 1) & (z_votes <= noise_threshold)
        processed[noise_mask] = 0

    return processed.astype(predictions.dtype)


def apply_edge_spanning_filter(
    predictions: np.ndarray,
    min_edge_fraction: float = 0.3,
    check_axes: tuple = (1, 2),
) -> np.ndarray:
    """Remove components that don't span from one edge to another.

    Real sheets span the entire H×W field of view. Components that don't
    reach opposite edges are likely noise or fragments. This filter checks
    if each connected component spans from one edge to the opposite edge
    in the H and/or W dimensions.

    Args:
        predictions: Binary predictions (0/1) with shape (D, H, W)
        min_edge_fraction: Minimum fraction of Z-slices where component touches edges
        check_axes: Which axes to check for spanning (1=H, 2=W)

    Returns:
        Filtered predictions with non-spanning components removed

    """
    labeled, num_features = ndimage.label(predictions)

    if num_features == 0:
        return predictions

    D, H, W = predictions.shape
    keep_mask = np.zeros_like(predictions, dtype=bool)

    for comp_id in range(1, num_features + 1):
        component = labeled == comp_id

        # Check if this component spans edges
        spans_edges = False

        for axis in check_axes:
            if axis == 1:  # H axis - check if touches both h=0 and h=H-1
                dim_size = H
                edge_low = component[:, 0, :]  # h=0 plane
                edge_high = component[:, -1, :]  # h=H-1 plane
            elif axis == 2:  # W axis - check if touches both w=0 and w=W-1
                dim_size = W
                edge_low = component[:, :, 0]  # w=0 plane
                edge_high = component[:, :, -1]  # w=W-1 plane
            else:
                continue

            # Count Z-slices where component touches both edges
            touches_low = edge_low.any(axis=-1) if edge_low.ndim > 1 else edge_low
            touches_high = edge_high.any(axis=-1) if edge_high.ndim > 1 else edge_high

            # For axis=1: touches_low/high are (D, W), reduce to (D,)
            # For axis=2: touches_low/high are (D, H), reduce to (D,)
            if touches_low.ndim > 1:
                touches_low = touches_low.any(axis=-1)
                touches_high = touches_high.any(axis=-1)

            # Check if component spans this axis in enough Z-slices
            spans_this_axis = touches_low & touches_high
            span_fraction = spans_this_axis.sum() / D

            if span_fraction >= min_edge_fraction:
                spans_edges = True
                break

        if spans_edges:
            keep_mask |= component

    return keep_mask.astype(predictions.dtype)


def apply_anisotropic_closing(
    predictions: np.ndarray,
    xy_iterations: int = 2,
    z_iterations: int = 1,
) -> np.ndarray:
    """Apply morphological closing with different strengths for XY vs Z.

    Since sheets are planar in XY but continuous in Z, we want stronger
    closing in XY (to fill holes) but gentler closing in Z (to avoid
    bridging separate sheets).

    Args:
        predictions: Binary predictions (0/1) with shape (D, H, W)
        xy_iterations: Number of closing iterations in XY plane
        z_iterations: Number of closing iterations along Z axis

    Returns:
        Processed predictions

    """
    from scipy.ndimage import binary_closing

    processed = predictions.copy()

    # XY closing: use structure that only connects in-plane
    # This fills holes within sheets without bridging along Z
    xy_structure = np.array(
        [
            [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
            [[0, 1, 0], [1, 1, 1], [0, 1, 0]],
            [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
        ],
        dtype=bool,
    )

    if xy_iterations > 0:
        processed = binary_closing(processed, structure=xy_structure, iterations=xy_iterations)

    # Z closing: use structure that only connects along Z
    # This maintains continuity along scroll length
    z_structure = np.array(
        [
            [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
            [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
            [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
        ],
        dtype=bool,
    )

    if z_iterations > 0:
        processed = binary_closing(processed, structure=z_structure, iterations=z_iterations)

    return processed.astype(predictions.dtype)


def apply_size_and_extent_filter(
    predictions: np.ndarray,
    min_voxels: int = 500,
    min_extent_ratio: float = 2.0,
) -> np.ndarray:
    """Remove small compact blobs while preserving thin extended sheets.

    Uses extent ratio = bbox_diagonal / volume^(1/3) to distinguish:
    - Thin sheets: high extent ratio (large bbox, small volume)
    - Compact blobs: low extent ratio (small bbox relative to volume)

    Args:
        predictions: Binary predictions (0/1)
        min_voxels: Minimum component size in voxels
        min_extent_ratio: Minimum extent ratio to keep (higher = more selective)
            - Compact cube has ratio ~1.7
            - Thin sheet 100x100x2 has ratio ~7.0
            - Recommend 2.0-3.0 for sheet filtering

    Returns:
        Filtered predictions with compact blobs removed

    """
    labeled, num_features = ndimage.label(predictions)

    if num_features == 0:
        return predictions

    keep_mask = np.zeros_like(predictions, dtype=bool)

    for comp_id in range(1, num_features + 1):
        component = labeled == comp_id
        voxel_count = component.sum()

        # Size filter
        if voxel_count < min_voxels:
            continue

        # Compute bounding box
        coords = np.where(component)
        if len(coords[0]) == 0:
            continue

        bbox_min = np.array([c.min() for c in coords])
        bbox_max = np.array([c.max() for c in coords])
        bbox_size = bbox_max - bbox_min + 1
        bbox_diagonal = np.sqrt(np.sum(bbox_size**2))

        # Compute extent ratio
        # Higher = more extended/thin, Lower = more compact
        volume_equiv_side = voxel_count ** (1 / 3)
        extent_ratio = bbox_diagonal / volume_equiv_side

        # Keep if extent ratio is high enough (thin structure) OR if large enough
        # Large components are likely real even if somewhat compact
        if extent_ratio >= min_extent_ratio or voxel_count >= min_voxels * 10:
            keep_mask |= component

    return keep_mask.astype(predictions.dtype)


def apply_cavity_filling(
    predictions: np.ndarray,
    max_cavity_size: int = 1000,
) -> np.ndarray:
    """Fill internal cavities within predicted structures.

    Cavities (enclosed background regions) create spurious k=2 Betti features.
    This fills small internal cavities while preserving the background.

    Args:
        predictions: Binary predictions (0/1)
        max_cavity_size: Maximum cavity size in voxels to fill

    Returns:
        Predictions with internal cavities filled

    """
    from scipy.ndimage import binary_fill_holes

    # Invert to find background
    background = predictions == 0

    # Label background components
    bg_labeled, num_bg = ndimage.label(background)

    if num_bg <= 1:
        # Only one background region = no cavities
        return predictions

    # Find the largest background component (the "outside")
    bg_sizes = ndimage.sum(background, bg_labeled, range(1, num_bg + 1))
    outside_label = np.argmax(bg_sizes) + 1

    # All other background components are potential cavities
    filled = predictions.copy()

    for bg_id in range(1, num_bg + 1):
        if bg_id == outside_label:
            continue

        cavity = bg_labeled == bg_id
        cavity_size = cavity.sum()

        if cavity_size <= max_cavity_size:
            # Fill this cavity
            filled[cavity] = 1

    return filled.astype(predictions.dtype)


def apply_gentle_edge_filter(
    predictions: np.ndarray,
    min_edge_touch_fraction: float = 0.1,
    min_component_size: int = 1000,
) -> np.ndarray:
    """Remove components that don't touch any edge of the volume.

    Less aggressive than edge-spanning filter. Only removes components
    that are completely internal (don't touch any boundary).

    Args:
        predictions: Binary predictions (0/1) with shape (D, H, W)
        min_edge_touch_fraction: Min fraction of Z-slices where component touches any edge
        min_component_size: Don't filter components larger than this

    Returns:
        Filtered predictions

    """
    labeled, num_features = ndimage.label(predictions)

    if num_features == 0:
        return predictions

    D, H, W = predictions.shape
    keep_mask = np.zeros_like(predictions, dtype=bool)

    for comp_id in range(1, num_features + 1):
        component = labeled == comp_id
        voxel_count = component.sum()

        # Always keep large components
        if voxel_count >= min_component_size:
            keep_mask |= component
            continue

        # Check if component touches any edge
        touches_edge = (
            component[:, 0, :].any()  # H=0 edge
            or component[:, -1, :].any()  # H=max edge
            or component[:, :, 0].any()  # W=0 edge
            or component[:, :, -1].any()  # W=max edge
            or component[0, :, :].any()  # D=0 edge
            or component[-1, :, :].any()  # D=max edge
        )

        if touches_edge:
            keep_mask |= component
        # else: component is internal, likely noise - remove it

    return keep_mask.astype(predictions.dtype)


def apply_sheet_merge_prevention(
    predictions: np.ndarray,
    min_separation: int = 3,
) -> np.ndarray:
    """Attempt to separate merged sheets using local thickness analysis.

    If predictions are thicker than expected and two sheets may have merged,
    try to find the separation boundary using distance transform valleys.

    Args:
        predictions: Binary predictions (0/1)
        min_separation: Minimum expected separation between sheets in voxels

    Returns:
        Predictions with potential sheet separations

    """
    from scipy.ndimage import distance_transform_edt

    if predictions.sum() == 0:
        return predictions

    # Compute distance transform
    dist = distance_transform_edt(predictions)

    # Find local maxima (potential sheet centers)
    # Sheets should have distance ~3-4 (half of 7-8 voxel thickness)
    max_expected_dist = 4.0

    # Find voxels that are too deep (potential merge regions)
    too_thick = dist > max_expected_dist

    if not too_thick.any():
        return predictions

    # Find valleys in the distance transform (potential separation lines)
    # Use Laplacian to find concave regions
    from scipy.ndimage import laplace

    lap = laplace(dist)

    # Positive Laplacian in thick regions suggests a merge point
    # These are local minima in the distance field within merged regions
    potential_splits = (lap > 0) & too_thick

    # Only split if it's a clear valley (surrounded by higher values)
    # This is conservative to avoid over-segmentation
    from scipy.ndimage import grey_erosion

    eroded_dist = grey_erosion(dist, size=3)
    is_local_min = (dist <= eroded_dist + 0.1) & potential_splits

    # Remove these split points
    result = predictions.copy()
    result[is_local_min] = 0

    return result.astype(predictions.dtype)


def apply_aggressive_connection(
    predictions: np.ndarray,
    dilation_radius: int = 3,
    closing_radius: int = 2,
) -> np.ndarray:
    """Aggressively connect nearby fragments using morphological operations.

    Uses dilation to bridge gaps, then erosion to restore approximate shape.
    This is a "closing" operation but with larger radius than standard.

    Args:
        predictions: Binary predictions (0/1)
        dilation_radius: Radius for initial dilation (bridges gaps up to 2*radius)
        closing_radius: Radius for final closing to smooth

    Returns:
        Predictions with nearby fragments connected

    """
    from scipy.ndimage import binary_closing, binary_dilation, binary_erosion

    if predictions.sum() == 0:
        return predictions

    # Create spherical structuring element
    struct_dilate = ndimage.generate_binary_structure(3, connectivity=1)

    # Dilate to connect nearby fragments
    dilated = binary_dilation(predictions, structure=struct_dilate, iterations=dilation_radius)

    # Erode back to approximate original size (but now connected)
    eroded = binary_erosion(dilated, structure=struct_dilate, iterations=dilation_radius)

    # Final closing to smooth
    if closing_radius > 0:
        struct_close = ndimage.generate_binary_structure(3, connectivity=1)
        result = binary_closing(eroded, structure=struct_close, iterations=closing_radius)
    else:
        result = eroded

    return result.astype(predictions.dtype)


def apply_fragment_removal(
    predictions: np.ndarray,
    min_fraction_of_largest: float = 0.01,
    min_absolute_size: int = 100,
) -> np.ndarray:
    """Remove small fragments relative to the largest component.

    Uses dynamic thresholding: keeps components that are at least
    min_fraction_of_largest times the size of the largest component.

    Args:
        predictions: Binary predictions (0/1)
        min_fraction_of_largest: Minimum size as fraction of largest component
        min_absolute_size: Absolute minimum size in voxels

    Returns:
        Predictions with small fragments removed

    """
    labeled, num_features = ndimage.label(predictions)

    if num_features == 0:
        return predictions

    # Get component sizes
    sizes = ndimage.sum(predictions, labeled, range(1, num_features + 1))
    max_size = sizes.max()

    # Dynamic threshold
    size_threshold = max(min_absolute_size, max_size * min_fraction_of_largest)

    # Keep only large enough components
    keep_mask = np.zeros_like(predictions, dtype=bool)
    for comp_id in range(1, num_features + 1):
        if sizes[comp_id - 1] >= size_threshold:
            keep_mask |= labeled == comp_id

    return keep_mask.astype(predictions.dtype)


def apply_keep_top_k_components(
    predictions: np.ndarray,
    k: int = 15,
    min_size: int = 100,
) -> np.ndarray:
    """Keep only the K largest connected components.

    Simple but effective: if we expect ~10-15 sheets in GT,
    keeping the top 15 components removes noise while preserving structure.

    Args:
        predictions: Binary predictions (0/1)
        k: Number of top components to keep
        min_size: Minimum size for a component to count

    Returns:
        Predictions with only top K components

    """
    labeled, num_features = ndimage.label(predictions)

    if num_features == 0:
        return predictions

    # Get component sizes
    sizes = np.array([ndimage.sum(predictions, labeled, i) for i in range(1, num_features + 1)])

    # Filter by minimum size first
    valid_indices = np.where(sizes >= min_size)[0]
    valid_sizes = sizes[valid_indices]

    if len(valid_sizes) == 0:
        return predictions

    # Get top K
    if len(valid_sizes) <= k:
        top_k_indices = valid_indices
    else:
        top_k_local = np.argsort(valid_sizes)[-k:]
        top_k_indices = valid_indices[top_k_local]

    # Create mask with only top K components
    keep_mask = np.zeros_like(predictions, dtype=bool)
    for idx in top_k_indices:
        comp_id = idx + 1  # labels are 1-indexed
        keep_mask |= labeled == comp_id

    return keep_mask.astype(predictions.dtype)


def apply_distance_based_merging(
    predictions: np.ndarray,
    max_gap: int = 5,
) -> np.ndarray:
    """Merge fragments that are within max_gap voxels of each other.

    Uses distance transform to find background voxels that are close
    to multiple foreground components and fills them to merge.

    Args:
        predictions: Binary predictions (0/1)
        max_gap: Maximum gap in voxels to bridge

    Returns:
        Predictions with nearby fragments merged

    """
    from scipy.ndimage import distance_transform_edt

    if predictions.sum() == 0:
        return predictions

    # Distance from each background voxel to nearest foreground
    dist_to_fg = distance_transform_edt(predictions == 0)

    # Find background voxels within max_gap of foreground
    potential_bridges = (predictions == 0) & (dist_to_fg <= max_gap / 2)

    # Dilate the prediction slightly and check if potential bridges connect components
    struct = ndimage.generate_binary_structure(3, connectivity=1)
    dilated = ndimage.binary_dilation(predictions, structure=struct, iterations=max_gap // 2)

    # The bridge points are where dilation reaches within the gap
    bridges = potential_bridges & dilated

    # Add bridges to prediction
    result = predictions | bridges

    return result.astype(predictions.dtype)


#################################################################################
# PRE-THRESHOLD PROBABILITY MANIPULATION FUNCTIONS
# These operate on probabilities [0,1] BEFORE binary thresholding
# Goal: Address fragmentation at the source by creating more connected probability maps
#################################################################################


def apply_probability_diffusion(
    probabilities: np.ndarray,
    sigma: float = 2.0,
    iterations: int = 3,
    preserve_peaks: bool = True,
) -> np.ndarray:
    """Diffuse high-probability regions to connect nearby uncertain areas.

    Uses anisotropic diffusion that spreads high confidence to neighboring
    low-confidence voxels, helping to connect fragmented predictions.

    Args:
        probabilities: Probability predictions [0, 1]
        sigma: Diffusion strength (higher = more spreading)
        iterations: Number of diffusion iterations
        preserve_peaks: If True, keep original values where they exceed diffused

    Returns:
        Diffused probabilities [0, 1]

    """
    result = probabilities.copy()

    for _ in range(iterations):
        # Gaussian blur to spread probabilities
        diffused = ndimage.gaussian_filter(result, sigma=sigma)

        if preserve_peaks:
            # Keep original high-confidence values
            result = np.maximum(result, diffused)
        else:
            result = diffused

    return np.clip(result, 0.0, 1.0)


def apply_hysteresis_threshold(
    probabilities: np.ndarray,
    high_thresh: float = 0.6,
    low_thresh: float = 0.3,
) -> np.ndarray:
    """Apply two-threshold hysteresis like Canny edge detection.

    High-confidence voxels (>high_thresh) are definite foreground.
    Low-confidence voxels (<low_thresh) are definite background.
    Voxels between thresholds are foreground if connected to high-confidence.

    This naturally creates more connected components by allowing
    uncertain regions to "inherit" confidence from neighbors.

    Args:
        probabilities: Probability predictions [0, 1]
        high_thresh: Threshold for definite foreground (seeds)
        low_thresh: Threshold for potential foreground (can join seeds)

    Returns:
        Binary predictions (0/1) as float32

    """
    from scipy.ndimage import label

    # Create seed mask (high confidence)
    high_mask = probabilities >= high_thresh

    # Create potential mask (could be foreground if connected to seeds)
    potential_mask = probabilities >= low_thresh

    # Use morphological reconstruction to find potential voxels connected to seeds
    # This is equivalent to flood-fill from high_mask within potential_mask
    from scipy.ndimage import binary_dilation

    # Iteratively dilate seeds within potential region
    result = high_mask.copy()
    struct = ndimage.generate_binary_structure(3, connectivity=1)

    while True:
        # Dilate current result
        dilated = binary_dilation(result, structure=struct)
        # Constrain to potential region
        new_result = dilated & potential_mask
        # Check for convergence
        if np.array_equal(new_result, result):
            break
        result = new_result

    return result.astype(np.float32)


def apply_watershed_extension(
    probabilities: np.ndarray,
    seed_thresh: float = 0.7,
    min_thresh: float = 0.3,
    compactness: float = 0.0,
) -> np.ndarray:
    """Extend high-confidence seeds using watershed segmentation.

    Uses high-confidence regions as seeds and grows them into
    uncertain regions using watershed algorithm. This creates
    more connected components by letting confident regions
    "claim" nearby uncertain voxels.

    Args:
        probabilities: Probability predictions [0, 1]
        seed_thresh: Threshold for watershed seeds
        min_thresh: Minimum probability to include in watershed
        compactness: Higher values give more regular shapes (0 = standard watershed)

    Returns:
        Binary predictions (0/1) as float32

    """
    from skimage.segmentation import watershed

    # Create seeds from high-confidence regions
    seeds_mask = probabilities >= seed_thresh
    labeled_seeds, n_seeds = ndimage.label(seeds_mask)

    if n_seeds == 0:
        # No seeds - fall back to simple threshold
        return (probabilities >= min_thresh).astype(np.float32)

    # Create mask of region to segment (above min_thresh)
    mask = probabilities >= min_thresh

    # Use negative probability as "elevation" so watershed grows high->low
    # (watershed normally grows from low to high elevation)
    elevation = 1.0 - probabilities

    # Run watershed from seeds within mask
    labels = watershed(
        elevation,
        markers=labeled_seeds,
        mask=mask,
        compactness=compactness,
    )

    # Convert to binary (any label > 0 is foreground)
    return (labels > 0).astype(np.float32)


def apply_graph_based_merging(
    predictions: np.ndarray,
    max_merge_distance: int = 10,
    target_components: int = 15,
    probability_map: np.ndarray | None = None,
) -> np.ndarray:
    """Merge nearby components using graph-based union-find.

    Builds a graph where nodes are connected components, edges connect
    components within max_merge_distance, and edge weights are based on
    proximity. Greedily merges closest pairs until target count reached.

    Args:
        predictions: Binary predictions (0/1)
        max_merge_distance: Maximum distance between components to consider merging
        target_components: Target number of components after merging
        probability_map: Optional probability map to weight merge decisions

    Returns:
        Binary predictions with components merged

    """
    from scipy.ndimage import distance_transform_edt

    labeled, num_features = ndimage.label(predictions)

    if num_features <= target_components:
        return predictions

    # Get component centroids and sizes
    component_info = []
    for comp_id in range(1, num_features + 1):
        mask = labeled == comp_id
        coords = np.where(mask)
        if len(coords[0]) == 0:
            continue
        centroid = np.array([c.mean() for c in coords])
        size = len(coords[0])
        component_info.append(
            {
                "id": comp_id,
                "centroid": centroid,
                "size": size,
                "mask": mask,
            },
        )

    # Build distance matrix between component surfaces
    # Use distance transform for efficiency
    n_comps = len(component_info)
    if n_comps <= target_components:
        return predictions

    # For each pair, find minimum surface-to-surface distance
    merge_candidates = []
    for i in range(n_comps):
        # Distance from background to component i's surface
        dist_i = distance_transform_edt(~component_info[i]["mask"])

        for j in range(i + 1, n_comps):
            # Minimum distance from component j's voxels to component i
            j_coords = np.where(component_info[j]["mask"])
            if len(j_coords[0]) == 0:
                continue

            # Sample points for efficiency if component is large
            n_sample = min(1000, len(j_coords[0]))
            if n_sample < len(j_coords[0]):
                indices = np.random.choice(len(j_coords[0]), n_sample, replace=False)
                sample_coords = tuple(c[indices] for c in j_coords)
            else:
                sample_coords = j_coords

            min_dist = dist_i[sample_coords].min()

            if min_dist <= max_merge_distance:
                merge_candidates.append(
                    {
                        "i": i,
                        "j": j,
                        "distance": min_dist,
                        "combined_size": component_info[i]["size"] + component_info[j]["size"],
                    },
                )

    # Sort by distance (closest first)
    merge_candidates.sort(key=lambda x: x["distance"])

    # Union-find structure
    parent = list(range(n_comps))

    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    def union(x, y):
        px, py = find(x), find(y)
        if px != py:
            parent[px] = py
            return True
        return False

    # Greedily merge until target reached
    current_count = n_comps
    for merge in merge_candidates:
        if current_count <= target_components:
            break
        if union(merge["i"], merge["j"]):
            current_count -= 1

    # Build merged prediction
    result = np.zeros_like(predictions)
    for i in range(n_comps):
        # Find root
        root = find(i)
        # All components with same root get merged
        if root == find(i):  # Only process each root once
            result |= component_info[i]["mask"]

    return result.astype(predictions.dtype)


def apply_connected_threshold(
    probabilities: np.ndarray,
    base_thresh: float = 0.3,
    connectivity_boost: float = 0.15,
    neighbor_radius: int = 3,
) -> np.ndarray:
    """Apply adaptive threshold based on local neighborhood connectivity.

    Voxels with high local mean probability get a lower effective threshold.
    This allows uncertain voxels near confident regions to join the prediction.

    Args:
        probabilities: Probability predictions [0, 1]
        base_thresh: Base threshold for isolated voxels
        connectivity_boost: Threshold reduction for well-connected voxels
        neighbor_radius: Radius for computing local mean

    Returns:
        Binary predictions (0/1) as float32

    """
    # Compute local mean probability
    from scipy.ndimage import uniform_filter

    size = 2 * neighbor_radius + 1
    local_mean = uniform_filter(probabilities, size=size, mode="reflect")

    # Adaptive threshold: lower where local mean is high
    # If local_mean is 1.0, threshold = base_thresh - connectivity_boost
    # If local_mean is 0.0, threshold = base_thresh
    adaptive_thresh = base_thresh - connectivity_boost * local_mean

    # Apply adaptive threshold
    result = probabilities >= adaptive_thresh

    return result.astype(np.float32)


def apply_probabilistic_closing(
    probabilities: np.ndarray,
    structure_size: int = 3,
    iterations: int = 2,
) -> np.ndarray:
    """Apply morphological closing in probability space.

    Instead of binary closing, this operates on probabilities:
    - Dilation: take local maximum
    - Erosion: take local minimum

    This fills gaps in probability maps before thresholding.

    Args:
        probabilities: Probability predictions [0, 1]
        structure_size: Size of structuring element
        iterations: Number of closing iterations

    Returns:
        Closed probabilities [0, 1]

    """
    from scipy.ndimage import maximum_filter, minimum_filter

    result = probabilities.copy()

    for _ in range(iterations):
        # Dilation (max filter)
        dilated = maximum_filter(result, size=structure_size)
        # Erosion (min filter)
        result = minimum_filter(dilated, size=structure_size)

    return np.clip(result, 0.0, 1.0)


#################################################################################
# Basic functions for loading predictions and computing metrics
#################################################################################


def load_img_and_label(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Load the image and label for a given id.

    For resized patches (resized=1), this function:
    1. Inverts any deterministic augmentation (flip/rotation) applied during data loading
    2. Resizes the patch back to original volume shape

    This ensures predictions align correctly with ground truth labels for metric computation.
    """
    # Ensure all label paths are the same for the given id
    if len(df["label_path"].unique()) != 1:
        raise ValueError(f"Multiple label paths found for dataframe with {len(df)} rows")

    # Load the label
    label = np.load(df["label_path"].iloc[0])
    shape = label.shape[0]

    # Initialize empty 3D cube and count array for averaging overlaps
    img = np.zeros((shape, shape, shape), dtype=np.float32)
    count = np.zeros((shape, shape, shape), dtype=np.int32)

    # Iterate through all sub-cubes in the dataframe
    for _, row in df.iterrows():
        # Load the sub-cube patch
        patch = np.load(row["patch_path"])

        # Ensure patch values are in valid probability range [0, 1]
        # Predictions should already be in this range from sigmoid, but clip for safety
        patch = np.clip(patch, 0.0, 1.0)

        # Get the coordinates for non-resized patches
        # NOTE: resized=0 patches have NO augmentation applied in validation mode
        # (random augmentation is only applied to training samples via augmentation_config)
        # So no inversion is needed here - predictions are already in original coordinates
        d0, h0, w0 = int(row["d0"]), int(row["h0"]), int(row["w0"])
        d1, h1, w1 = int(row["d1"]), int(row["h1"]), int(row["w1"])

        # Expected shape based on coordinates
        expected_shape = (d1 - d0, h1 - h0, w1 - w0)

        # Verify coordinates are within volume bounds
        if d1 > shape or h1 > shape or w1 > shape:
            warnings.warn(
                f"Patch coordinates exceed volume bounds. "
                f"Volume shape: ({shape}, {shape}, {shape}), "
                f"Patch coords: d[{d0}:{d1}], h[{h0}:{h1}], w[{w0}:{w1}]. "
                f"Skipping patch.",
                stacklevel=2,
            )
            continue

        # Handle shape mismatch by cropping/padding the patch to expected size
        # This occurs when predictions are saved at a different resolution than
        # what the coordinates specify (e.g., 256x256 patches vs 320x320 coordinates)
        if patch.shape != expected_shape:
            # If patch is smaller than expected, pad symmetrically
            if (
                patch.shape[0] < expected_shape[0]
                or patch.shape[1] < expected_shape[1]
                or patch.shape[2] < expected_shape[2]
            ):
                warnings.warn(
                    f"Patch {row['patch_path']} is smaller than expected. "
                    f"Patch shape: {patch.shape}, Expected: {expected_shape}. "
                    f"Padding with edge values for smooth blending.",
                    stacklevel=2,
                )
                # Calculate padding needed on each side
                pad_d = max(0, expected_shape[0] - patch.shape[0])
                pad_h = max(0, expected_shape[1] - patch.shape[1])
                pad_w = max(0, expected_shape[2] - patch.shape[2])

                # Use edge padding to avoid introducing artificial zeros
                patch = np.pad(
                    patch,
                    ((0, pad_d), (0, pad_h), (0, pad_w)),
                    mode="edge",
                )

            # If patch is larger than expected, crop from center for better alignment
            if (
                patch.shape[0] > expected_shape[0]
                or patch.shape[1] > expected_shape[1]
                or patch.shape[2] > expected_shape[2]
            ):
                # Calculate center crop offsets
                d_offset = max(0, (patch.shape[0] - expected_shape[0]) // 2)
                h_offset = max(0, (patch.shape[1] - expected_shape[1]) // 2)
                w_offset = max(0, (patch.shape[2] - expected_shape[2]) // 2)

                # Crop from center
                patch = patch[
                    d_offset : d_offset + expected_shape[0],
                    h_offset : h_offset + expected_shape[1],
                    w_offset : w_offset + expected_shape[2],
                ]

            # Verify the resize worked
            if patch.shape != expected_shape:
                raise ValueError(
                    f"Failed to resize patch to expected shape. "
                    f"After resize: {patch.shape}, Expected: {expected_shape}",
                )

        # Add the patch to the accumulator
        img[d0:d1, h0:h1, w0:w1] += patch
        count[d0:d1, h0:h1, w0:w1] += 1

    # Verify that every voxel has at least 1 counts:
    # - 1 from the resized full volume
    # - 1+ from the patches that cover the volume
    min_count = count.min()
    if min_count < 1:
        num_low_voxels = np.sum(count < 1)
        total_voxels = count.size
        warnings.warn(
            f"Expected every voxel to have count >= 2 (resized + patches), "
            f"but found min count = {min_count}. "
            f"{num_low_voxels}/{total_voxels} voxels have count < 2. "
            "Check that patches fully cover the volume and resized prediction exists.",
            stacklevel=2,
        )

    # Average the overlapping voxels
    # Avoid division by zero (though count should always be >= 1 where img is non-zero)
    mask = count > 0
    img[mask] /= count[mask]

    # Ensure final predictions are in valid [0, 1] range after averaging
    # This handles any numerical precision issues from accumulation
    img = np.clip(img, 0.0, 1.0)

    # Mask out voxels in label where no sub-cubes were used (count == 0)
    label[~mask] = 2

    return img, label


def _compute_scores_for_id_wrapper(args: tuple) -> tuple[int, dict]:
    """Wrapper function for multiprocessing Pool.

    Takes a single tuple argument to work with Pool.imap_unordered.
    """
    (
        img_id,
        img_df_dict,  # Pass as dict to avoid pickle issues with DataFrame
        topo_weight,
        surface_dice_weight,
        voi_weight,
        surface_tolerance,
        voi_connectivity,
        voi_transform,
        voi_alpha,
        postprocess_config,  # New parameter for post-processing options
    ) = args

    try:
        # Reconstruct DataFrame from dict
        img_df = pd.DataFrame(img_df_dict)

        pr, gt = load_img_and_label(img_df)

        # Apply optional CLAHE (before smoothing)
        if postprocess_config.get("apply_clahe", False):
            clip_limit = postprocess_config.get("clahe_clip_limit", 0.03)
            pr = apply_adaptive_histogram_equalization(pr, clip_limit=clip_limit)

        # Apply optional Gaussian smoothing before thresholding
        if postprocess_config.get("gaussian_smoothing", False):
            sigma = postprocess_config.get("gaussian_sigma", 0.5)
            pr = apply_gaussian_smoothing(pr, sigma=sigma)

        # === PRE-THRESHOLD PROBABILITY MANIPULATION ===
        # These operate on probability maps to reduce fragmentation BEFORE thresholding

        # Apply probability diffusion to spread confidence
        if postprocess_config.get("probability_diffusion", False):
            pr = apply_probability_diffusion(
                pr,
                sigma=postprocess_config.get("diffusion_sigma", 2.0),
                iterations=postprocess_config.get("diffusion_iterations", 3),
                preserve_peaks=postprocess_config.get("diffusion_preserve_peaks", True),
            )

        # Apply probabilistic closing (max/min filters on probabilities)
        if postprocess_config.get("probabilistic_closing", False):
            pr = apply_probabilistic_closing(
                pr,
                structure_size=postprocess_config.get("prob_closing_size", 3),
                iterations=postprocess_config.get("prob_closing_iterations", 2),
            )

        # Apply hysteresis threshold (Canny-style two-threshold)
        # NOTE: This produces binary output, so skip the normal threshold
        if postprocess_config.get("hysteresis_threshold", False):
            pr = apply_hysteresis_threshold(
                pr,
                high_thresh=postprocess_config.get("hysteresis_high", 0.6),
                low_thresh=postprocess_config.get("hysteresis_low", 0.3),
            )
            # Skip normal thresholding since hysteresis already produces binary
            pr = pr.astype(np.uint8)
        # Apply watershed extension from high-confidence seeds
        # NOTE: This produces binary output, so skip the normal threshold
        elif postprocess_config.get("watershed_extension", False):
            pr = apply_watershed_extension(
                pr,
                seed_thresh=postprocess_config.get("watershed_seed_thresh", 0.7),
                min_thresh=postprocess_config.get("watershed_min_thresh", 0.3),
                compactness=postprocess_config.get("watershed_compactness", 0.0),
            )
            # Skip normal thresholding since watershed already produces binary
            pr = pr.astype(np.uint8)
        # Apply connected threshold (adaptive based on local mean)
        # NOTE: This produces binary output, so skip the normal threshold
        elif postprocess_config.get("connected_threshold", False):
            pr = apply_connected_threshold(
                pr,
                base_thresh=postprocess_config.get("connected_base_thresh", 0.3),
                connectivity_boost=postprocess_config.get("connected_boost", 0.15),
                neighbor_radius=postprocess_config.get("connected_radius", 3),
            )
            # Skip normal thresholding since connected threshold already produces binary
            pr = pr.astype(np.uint8)
        else:
            # Apply threshold to create binary mask
            # Support custom threshold instead of fixed 0.5
            threshold = postprocess_config.get("binary_threshold", BINARY_THRESHOLD)
            pr = (pr > threshold).astype(np.uint8)

        # Apply optional connected component refinement
        if postprocess_config.get("component_refinement", False):
            percentile = postprocess_config.get("component_percentile", 5.0)
            pr = apply_connected_component_refinement(pr, min_size_percentile=percentile)

        # Apply optional morphological post-processing
        if postprocess_config.get("morphological_cleanup", False):
            pr = apply_morphological_postprocessing(
                pr,
                remove_small_objects_threshold=postprocess_config.get("remove_small_objects", 50),
                remove_small_holes_threshold=postprocess_config.get("remove_small_holes", 50),
                apply_closing=postprocess_config.get("apply_closing", True),
                closing_iterations=postprocess_config.get("closing_iterations", 1),
            )

        # === NEW SHEET-AWARE POST-PROCESSING ===

        # Apply directional hole filling (2D closing per Z-slice)
        if postprocess_config.get("directional_hole_filling", False):
            pr = apply_directional_hole_filling(
                pr,
                max_hole_size=postprocess_config.get("max_hole_size", 100),
                axis=postprocess_config.get("hole_fill_axis", 0),
            )

        # Apply anisotropic closing (stronger in XY than Z)
        if postprocess_config.get("anisotropic_closing", False):
            pr = apply_anisotropic_closing(
                pr,
                xy_iterations=postprocess_config.get("xy_closing_iterations", 2),
                z_iterations=postprocess_config.get("z_closing_iterations", 1),
            )

        # Apply Z-continuity enforcement
        if postprocess_config.get("z_continuity", False):
            pr = apply_z_continuity_enforcement(
                pr,
                window_size=postprocess_config.get("z_window_size", 5),
                threshold=postprocess_config.get("z_threshold", 0.6),
                fill_gaps=postprocess_config.get("z_fill_gaps", True),
                remove_noise=postprocess_config.get("z_remove_noise", True),
            )

        # Apply sheet thinning
        if postprocess_config.get("sheet_thinning", False):
            pr = apply_sheet_thinning(
                pr,
                target_thickness=postprocess_config.get("target_thickness", 6),
                min_thickness=postprocess_config.get("min_thickness", 4),
            )

        # Apply edge-spanning filter (remove non-spanning components)
        if postprocess_config.get("edge_spanning_filter", False):
            pr = apply_edge_spanning_filter(
                pr,
                min_edge_fraction=postprocess_config.get("min_edge_fraction", 0.3),
                check_axes=postprocess_config.get("check_axes", (1, 2)),
            )

        # === PRECISION-FOCUSED POST-PROCESSING ===

        # Apply gentle edge filter (remove internal floating components)
        if postprocess_config.get("gentle_edge_filter", False):
            pr = apply_gentle_edge_filter(
                pr,
                min_edge_touch_fraction=postprocess_config.get("min_edge_touch_fraction", 0.1),
                min_component_size=postprocess_config.get("edge_filter_min_size", 1000),
            )

        # Apply size and extent filter (remove compact blobs)
        if postprocess_config.get("size_extent_filter", False):
            pr = apply_size_and_extent_filter(
                pr,
                min_voxels=postprocess_config.get("min_voxels", 500),
                min_extent_ratio=postprocess_config.get("min_extent_ratio", 2.0),
            )

        # Apply cavity filling (reduce k=2 Betti errors)
        if postprocess_config.get("cavity_filling", False):
            pr = apply_cavity_filling(
                pr,
                max_cavity_size=postprocess_config.get("max_cavity_size", 1000),
            )

        # Apply sheet merge prevention
        if postprocess_config.get("sheet_merge_prevention", False):
            pr = apply_sheet_merge_prevention(
                pr,
                min_separation=postprocess_config.get("min_separation", 3),
            )

        # === ANTI-FRAGMENTATION POST-PROCESSING ===

        # Apply aggressive connection (bridge gaps between fragments)
        if postprocess_config.get("aggressive_connection", False):
            pr = apply_aggressive_connection(
                pr,
                dilation_radius=postprocess_config.get("dilation_radius", 3),
                closing_radius=postprocess_config.get("closing_radius", 2),
            )

        # Apply distance-based merging
        if postprocess_config.get("distance_merging", False):
            pr = apply_distance_based_merging(
                pr,
                max_gap=postprocess_config.get("max_gap", 5),
            )

        # Apply fragment removal (remove small disconnected pieces)
        if postprocess_config.get("fragment_removal", False):
            pr = apply_fragment_removal(
                pr,
                min_fraction_of_largest=postprocess_config.get("min_fraction", 0.01),
                min_absolute_size=postprocess_config.get("min_absolute_size", 100),
            )

        # Keep only top K components
        if postprocess_config.get("keep_top_k", False):
            pr = apply_keep_top_k_components(
                pr,
                k=postprocess_config.get("top_k", 15),
                min_size=postprocess_config.get("top_k_min_size", 100),
            )

        # Apply graph-based merging (union-find to merge nearby components)
        if postprocess_config.get("graph_merging", False):
            pr = apply_graph_based_merging(
                pr,
                max_merge_distance=postprocess_config.get("merge_distance", 10),
                target_components=postprocess_config.get("target_components", 15),
            )

        # Kaggle metric
        score_report = compute_leaderboard_score(
            predictions=pr,
            labels=gt,
            dims=(0, 1, 2),
            spacing=(1.0, 1.0, 1.0),  # (z, y, x)
            surface_tolerance=surface_tolerance,
            voi_connectivity=voi_connectivity,
            voi_transform=voi_transform,
            voi_alpha=voi_alpha,
            combine_weights=(topo_weight, surface_dice_weight, voi_weight),
            fg_threshold=None,  # None => legacy "!= 0"
            ignore_label=2,  # Voxels with this GT label are ignored
            ignore_mask=None,
        )

        result = {
            "surface_dice": score_report.surface_dice,
            "topo_score": score_report.topo.toposcore,
            "voi_score": score_report.voi.voi_score,
            "kaggle_score": score_report.score,
        }

        # Explicitly free memory
        del pr, gt, score_report
        gc.collect()

        return img_id, result

    except Exception as e:
        # Return error info instead of raising - allows other workers to continue
        return img_id, {"error": str(e)}


def compute_average_metrics(
    data_dir: Path,
    base_dir: Path,
    topo_weight: float = DEFAULT_TOPO_WEIGHT,
    surface_dice_weight: float = DEFAULT_SURFACE_DICE_WEIGHT,
    voi_weight: float = DEFAULT_VOI_WEIGHT,
    surface_tolerance: float = DEFAULT_SURFACE_TOLERANCE,
    voi_connectivity: int = DEFAULT_VOI_CONNECTIVITY,
    voi_transform: str = DEFAULT_VOI_TRANSFORM,
    voi_alpha: float = DEFAULT_VOI_ALPHA,
    max_workers: int | None = None,
    use_sequential: bool = False,
    postprocess_config: dict | None = None,
) -> tuple[float, float, float, float]:
    """Compute average metrics across all images.

    Uses multiprocessing Pool with maxtasksperchild=1 for memory safety,
    or sequential processing if use_sequential=True.

    Args:
        data_dir: Directory containing prediction .npy files and val_patches.csv
        base_dir: Base directory of the project
        topo_weight: Weight for topological score
        surface_dice_weight: Weight for surface dice score
        voi_weight: Weight for VOI score
        surface_tolerance: Tolerance for surface distance
        voi_connectivity: Connectivity for VOI calculation
        voi_transform: Transform function for VOI
        voi_alpha: Alpha parameter for VOI
        max_workers: Maximum number of parallel workers.
            Default: min(4, cpu_count // 4, num_images)
        use_sequential: If True, process images sequentially (no multiprocessing)
        postprocess_config: Dictionary with post-processing options:
            - gaussian_smoothing (bool): Apply Gaussian smoothing before threshold
            - gaussian_sigma (float): Sigma for Gaussian smoothing (default: 0.5)
            - morphological_cleanup (bool): Apply morphological operations
            - remove_small_objects (int): Min object size in voxels (default: 50)
            - remove_small_holes (int): Max hole size to fill (default: 50)
            - apply_closing (bool): Apply morphological closing (default: True)
            - closing_iterations (int): Number of closing iterations (default: 1)

    Returns:
        Tuple of (avg_surface_dice, avg_topo_score, avg_voi_score, avg_kaggle_score)

    """
    # Default post-processing config
    if postprocess_config is None:
        postprocess_config = {}
    # Load the dataframe of validation patches
    val_df = pd.read_csv(data_dir / "val_patches.csv")

    # Read tmp numpy files for prediction data
    tmp_info = []
    npy_files = list(data_dir.glob("*.npy"))
    if not npy_files:
        warnings.warn(
            f"No .npy prediction files found in {data_dir}. "
            "Returning zero scores. Check that predictions are being saved correctly.",
            stacklevel=2,
        )
        return 0.0, 0.0, 0.0, 0.0

    for file in npy_files:
        try:
            idx = int(file.stem)
            ds = val_df.iloc[idx]
            img_id = int(ds.id)
            tmp_info.append(
                {
                    "id": img_id,
                    "d0": int(ds.d0),
                    "h0": int(ds.h0),
                    "w0": int(ds.w0),
                    "d1": int(ds.d1),
                    "h1": int(ds.h1),
                    "w1": int(ds.w1),
                    "patch_path": str(file),
                    "label_path": str(
                        base_dir / "data/raw/kaggle_converted/train_labels" / f"{img_id}.npy",
                    ),
                    "img_shape": ds.get("img_shape"),
                },
            )
        except (ValueError, IndexError) as e:
            warnings.warn(
                f"Could not parse filename {file.name}: {e}. Skipping.",
                stacklevel=2,
            )
            continue

    if not tmp_info:
        warnings.warn(
            f"Failed to parse any .npy files in {data_dir}. Returning zero scores. "
            "Check that prediction files are named correctly.",
            stacklevel=2,
        )
        return 0.0, 0.0, 0.0, 0.0

    tmp_df = pd.DataFrame(tmp_info)
    tmp_df = tmp_df.sort_values(by=["id", "d0", "h0", "w0"]).reset_index(drop=True)

    # unique_ids = tmp_df["id"].unique()[0:10]
    unique_ids = tmp_df["id"].unique()
    n_images = len(unique_ids)
    print(f"Computing Kaggle metrics for {n_images} unique images...")

    # Prepare task arguments - convert DataFrames to dicts for pickling
    tasks = []
    for img_id in unique_ids:
        img_df = tmp_df[tmp_df["id"] == img_id]
        tasks.append(
            (
                img_id,
                img_df.to_dict("records"),  # Convert to list of dicts for clean pickling
                topo_weight,
                surface_dice_weight,
                voi_weight,
                surface_tolerance,
                voi_connectivity,
                voi_transform,
                voi_alpha,
                postprocess_config,  # Pass post-processing config
            ),
        )

    scores = {}
    failed_ids = []

    if use_sequential:
        # Sequential processing - useful for debugging
        print("Using sequential processing...")
        for i, task in enumerate(tasks):
            img_id, result = _compute_scores_for_id_wrapper(task)
            if "error" in result:
                print(f"Warning: Failed ID {img_id}: {result['error']}")
                failed_ids.append(img_id)
                scores[img_id] = {
                    "surface_dice": 0.0,
                    "topo_score": 0.0,
                    "voi_score": 0.0,
                    "kaggle_score": 0.0,
                }
            else:
                scores[img_id] = result

            if (i + 1) % max(1, n_images // 10) == 0 or (i + 1) == n_images:
                print(f"  Completed {i + 1}/{n_images} images...")
    else:
        # Parallel processing using multiprocessing.Pool
        # This is more stable than ProcessPoolExecutor for worker recycling
        if max_workers is None:
            max_workers = max(1, min(4, cpu_count() // 4, n_images))

        print(f"Using {max_workers} parallel workers...")

        # Use Pool with maxtasksperchild=1 to force worker recycling
        # This prevents memory accumulation and potential crashes
        with _MP_CONTEXT.Pool(
            processes=max_workers,
            initializer=_worker_init,
            maxtasksperchild=1,
        ) as pool:
            # imap_unordered processes results as they complete
            completed = 0
            for img_id, result in pool.imap_unordered(
                _compute_scores_for_id_wrapper,
                tasks,
                chunksize=1,  # Process one at a time for proper worker recycling
            ):
                if "error" in result:
                    print(f"Warning: Failed ID {img_id}: {result['error']}")
                    failed_ids.append(img_id)
                    scores[img_id] = {
                        "surface_dice": 0.0,
                        "topo_score": 0.0,
                        "voi_score": 0.0,
                        "kaggle_score": 0.0,
                    }
                else:
                    scores[img_id] = result

                completed += 1
                if completed % max(1, n_images // 10) == 0 or completed == n_images:
                    print(f"  Completed {completed}/{n_images} images...")

    if failed_ids:
        print(f"Warning: {len(failed_ids)} images failed to compute metrics")

    if not scores:
        warnings.warn("No scores computed successfully. Returning zeros.", stacklevel=2)
        return 0.0, 0.0, 0.0, 0.0

    # Average all scores
    avg_surface_dice = np.mean([s["surface_dice"] for s in scores.values()])
    avg_topo_score = np.mean([s["topo_score"] for s in scores.values()])
    avg_voi_score = np.mean([s["voi_score"] for s in scores.values()])
    avg_kaggle_score = np.mean([s["kaggle_score"] for s in scores.values()])

    print(f"Kaggle metrics complete: score={avg_kaggle_score:.4f}")

    return avg_surface_dice, avg_topo_score, avg_voi_score, avg_kaggle_score
