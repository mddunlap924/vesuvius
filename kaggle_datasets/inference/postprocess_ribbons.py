"""Advanced post-processing techniques for 3D thin ribbon segmentation.

DATA-DRIVEN INSIGHTS FROM GT ANALYSIS (200 volumes, 5220 ribbons):
- Real ribbons are ~8-9 voxels wide (diameter), not 3
- Ribbons extend primarily along z-axis (mean z-span 273 slices, 75% full Z)
- Ribbons do NOT touch XY borders (0% in GT) - border enforcement disabled
- Very few holes in ribbons (0.3%)
- High z-continuity (0.92 IoU between adjacent slices)
- Many tiny fragments (<100 voxels) are noise, real ribbons are large
- ~45% of volume is ignore/mask regions (label=2)

Post-processing strategies implemented:
1. Adaptive thresholding based on local statistics
2. Morphological hole filling (per z-slice and 3D)
3. Connected component analysis with size filtering
4. Z-axis continuity enforcement
5. Border connectivity enforcement (disabled by default based on GT)
6. Ribbon separation (watershed-based)
7. Thin structure preservation (skeletonization + dilation)
8. Hysteresis thresholding
"""

import warnings
from collections.abc import Callable
from typing import Any, Dict, Optional

import numpy as np
from scipy import ndimage
from scipy.ndimage import (
    binary_dilation,
    binary_erosion,
    binary_fill_holes,
    distance_transform_edt,
    gaussian_filter,
)
from scipy.ndimage import label as scipy_label
from skimage.filters import threshold_local, threshold_otsu
from skimage.measure import regionprops
from skimage.morphology import (
    ball,
    disk,
    remove_small_holes,
    remove_small_objects,
    skeletonize,  # Handles both 2D and 3D in newer versions
)
from skimage.morphology import (
    closing as morph_closing,
)
from skimage.morphology import (
    opening as morph_opening,
)
from skimage.segmentation import watershed

# Alias for backwards compatibility
skeletonize_3d = skeletonize

# Suppress specific deprecation warnings from skimage
warnings.filterwarnings("ignore", message=".*min_size.*deprecated.*")
warnings.filterwarnings("ignore", message=".*area_threshold.*deprecated.*")


def compute_surfaceness_simple(
    volume: np.ndarray,
    sigma: float = 1.0,
) -> np.ndarray:
    """Compute a simple surfaceness measure using structure tensor.

    For sheet-like structures, we want regions where intensity varies
    strongly in one direction but not in the other two.

    This is a simplified approach that's faster than full Hessian eigenvalue
    computation.

    Args:
        volume: 3D input volume (probability map)
        sigma: Gaussian smoothing sigma

    Returns:
        Surfaceness response (0-1 range)

    """
    # Smooth the volume
    smoothed = gaussian_filter(volume.astype(np.float64), sigma=sigma)

    # Compute gradients
    gz, gy, gx = np.gradient(smoothed)

    # Gradient magnitude - high at edges/surfaces
    grad_mag = np.sqrt(gx**2 + gy**2 + gz**2)

    # For ribbons extending along Z, we expect low gz but higher gx, gy
    # Compute ratio of in-plane gradient to total
    in_plane_grad = np.sqrt(gx**2 + gy**2)

    # Surfaceness: high gradient magnitude with primarily in-plane direction
    with np.errstate(divide="ignore", invalid="ignore"):
        surfaceness = in_plane_grad / (grad_mag + 1e-10)

    # Weight by gradient magnitude to suppress flat regions
    surfaceness = surfaceness * np.tanh(grad_mag * 10)

    # Smooth the result
    surfaceness = gaussian_filter(surfaceness, sigma=sigma / 2)

    # Normalize to 0-1
    if surfaceness.max() > 0:
        surfaceness = (surfaceness - surfaceness.min()) / (
            surfaceness.max() - surfaceness.min() + 1e-10
        )

    return surfaceness.astype(np.float32)


def frangi_surfaceness_filter(
    volume: np.ndarray,
    sigma: float = 1.0,
    dark_on_bright: bool = False,
) -> np.ndarray:
    """Modified Frangi filter that enhances surfaceness (sheet-like structures).

    For surfaces/sheets, we look for structures where intensity changes sharply
    in one direction (across the surface) but remains constant in two directions
    (along the surface).

    This simplified version uses Laplacian-based detection which is faster
    and more robust than full Hessian eigenvalue decomposition.

    Args:
        volume: 3D probability/intensity volume
        sigma: Scale for smoothing
        dark_on_bright: If True, detect dark sheets on bright background

    Returns:
        Surfaceness response volume (0-1 range)

    """
    # Smooth volume
    smoothed = gaussian_filter(volume.astype(np.float64), sigma=sigma)

    # Compute second derivatives
    gz, gy, gx = np.gradient(smoothed)
    gzz, _, _ = np.gradient(gz)
    _, gyy, _ = np.gradient(gy)
    _, _, gxx = np.gradient(gx)

    # Laplacian = trace of Hessian
    laplacian = gxx + gyy + gzz

    # For bright sheets on dark background, laplacian should be negative
    # (concave down at the ridge)
    if dark_on_bright:
        response = laplacian
    else:
        response = -laplacian

    # Only keep positive responses (sheet-like)
    response = np.maximum(response, 0)

    # Normalize
    if response.max() > 0:
        response = response / response.max()

    return response.astype(np.float32)


def surfaceness_threshold(
    pr: np.ndarray,
    threshold: float = 0.75,
    sigma: float = 1.0,
    final_threshold: float = 0.5,
) -> np.ndarray:
    """Apply surfaceness-enhanced thresholding.

    Based on the approach: threshold softmax at 0.75, then apply modified
    Frangi filter to enhance sheet-like (ribbon) structures.

    Args:
        pr: Probability map (float32, 0-1)
        threshold: Initial high-confidence threshold (default 0.75)
        sigma: Scale for surfaceness computation
        final_threshold: Threshold for final binary mask

    Returns:
        Binary mask (uint8)

    """
    # Step 1: Get high-confidence seed regions
    high_conf = (pr > threshold).astype(np.float32)

    # Step 2: Compute surfaceness on the probability map
    surfaceness = frangi_surfaceness_filter(pr, sigma=sigma)

    # Step 3: Combine - use surfaceness to expand from high-confidence seeds
    # The surfaceness helps recover ribbon regions that might be just below threshold
    combined = pr * (1.0 + surfaceness)  # Boost probabilities by surfaceness

    # Step 4: Use lower threshold on enhanced probabilities
    # Regions with high surfaceness get boosted, making them easier to threshold
    result = (combined > final_threshold).astype(np.uint8)

    # Step 5: Ensure we keep all high-confidence regions
    result = np.maximum(result, high_conf.astype(np.uint8))

    return result


def simple_threshold(pr: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """Baseline simple thresholding."""
    return (pr > threshold).astype(np.uint8)


def hysteresis_threshold(
    pr: np.ndarray,
    low_threshold: float = 0.15,
    high_threshold: float = 0.35,
) -> np.ndarray:
    """Hysteresis thresholding - keeps regions connected to high-confidence areas.

    Pixels above high_threshold are definitely foreground.
    Pixels between low and high are foreground only if connected to high pixels.
    This reduces noise while preserving weak but connected ribbon regions.
    """
    high_mask = pr > high_threshold
    low_mask = pr > low_threshold

    # Label connected components in the low threshold mask
    labeled, num_features = scipy_label(low_mask)

    # Find which components contain high-confidence pixels
    high_labels = np.unique(labeled[high_mask])
    high_labels = high_labels[high_labels != 0]  # Remove background

    # Keep only components that have high-confidence pixels
    result = np.isin(labeled, high_labels).astype(np.uint8)
    return result


def adaptive_threshold_3d(
    pr: np.ndarray,
    block_size: int = 15,
    offset: float = 0.05,
) -> np.ndarray:
    """Adaptive thresholding that adjusts threshold locally.

    Useful when probability values vary across the volume.
    Applies adaptive thresholding per z-slice then combines.
    """
    result = np.zeros_like(pr, dtype=np.uint8)

    for z in range(pr.shape[0]):
        slice_2d = pr[z]
        if slice_2d.max() - slice_2d.min() < 0.01:
            # Skip nearly uniform slices
            continue

        # Normalize slice to 0-1 range for consistent thresholding
        slice_norm = (slice_2d - slice_2d.min()) / (slice_2d.max() - slice_2d.min() + 1e-8)

        # Apply local adaptive threshold
        local_thresh = threshold_local(slice_norm, block_size, offset=offset)
        result[z] = (slice_norm > local_thresh).astype(np.uint8)

    return result


def fill_holes_2d_per_slice(mask: np.ndarray) -> np.ndarray:
    """Fill holes in each z-slice independently.

    Enforces the constraint that ribbons have no holes.
    Works slice-by-slice to preserve ribbon structure along z.
    """
    result = np.zeros_like(mask)
    for z in range(mask.shape[0]):
        result[z] = binary_fill_holes(mask[z]).astype(mask.dtype)
    return result


def fill_holes_3d(mask: np.ndarray, max_hole_size: int = 100) -> np.ndarray:
    """Fill small 3D holes in the mask.

    Uses remove_small_holes from skimage to fill holes up to max_hole_size.
    """
    return remove_small_holes(mask.astype(bool), area_threshold=max_hole_size).astype(np.uint8)


def remove_small_components(
    mask: np.ndarray,
    min_size: int = 50,
    connectivity: int = 3,
) -> np.ndarray:
    """Remove small disconnected components.

    Removes noise and small artifacts that are unlikely to be real ribbons.
    """
    return remove_small_objects(
        mask.astype(bool),
        min_size=min_size,
        connectivity=connectivity,
    ).astype(np.uint8)


def enforce_z_continuity(
    mask: np.ndarray,
    min_z_span: int = 10,
    min_overlap_ratio: float = 0.3,
) -> np.ndarray:
    """Enforce that ribbons extend consistently along z-axis.

    Removes components that don't span enough z-slices or have
    poor z-continuity (large gaps or inconsistent overlap between slices).
    """
    labeled, num_features = scipy_label(mask)
    result = np.zeros_like(mask)

    for i in range(1, num_features + 1):
        component = labeled == i

        # Check z-span
        z_indices = np.where(component.any(axis=(1, 2)))[0]
        if len(z_indices) < min_z_span:
            continue

        # Check z-continuity (overlap between consecutive slices)
        good_continuity = True
        for z_idx in range(len(z_indices) - 1):
            z1, z2 = z_indices[z_idx], z_indices[z_idx + 1]
            if z2 - z1 > 2:  # Gap larger than 2 slices
                good_continuity = False
                break

            # Check overlap ratio
            slice1 = component[z1]
            slice2 = component[z2]
            intersection = np.logical_and(slice1, slice2).sum()
            union = np.logical_or(slice1, slice2).sum()
            if union > 0 and intersection / union < min_overlap_ratio:
                good_continuity = False
                break

        if good_continuity:
            result[component] = 1

    return result


def enforce_border_connectivity(
    mask: np.ndarray,
    border_margin: int = 5,
) -> np.ndarray:
    """Keep only ribbons that start and end at xy-plane borders.

    Since ribbons extend from border to border in xy, we filter out
    components that don't touch the borders.
    """
    labeled, num_features = scipy_label(mask)
    result = np.zeros_like(mask)

    _, h, w = mask.shape

    for i in range(1, num_features + 1):
        component = labeled == i

        # Project onto xy plane
        xy_projection = component.any(axis=0)

        # Check if it touches borders
        touches_top = xy_projection[:border_margin, :].any()
        touches_bottom = xy_projection[-border_margin:, :].any()
        touches_left = xy_projection[:, :border_margin].any()
        touches_right = xy_projection[:, -border_margin:].any()

        # Ribbon should touch at least two opposite or adjacent borders
        border_count = sum([touches_top, touches_bottom, touches_left, touches_right])
        if border_count >= 2:
            result[component] = 1

    return result


def separate_touching_ribbons(
    mask: np.ndarray,
    pr: np.ndarray,
    min_distance: int = 2,
) -> np.ndarray:
    """Separate ribbons that might be incorrectly merged.

    Uses watershed segmentation on the distance transform, guided by
    the probability map to find natural separation points.
    """
    if mask.sum() == 0:
        return mask

    # Distance transform from background
    distance = distance_transform_edt(mask)

    # Find local maxima as markers (ribbon centers)
    # Use erosion to find ridge centers
    from scipy.ndimage import maximum_filter

    local_max = (distance == maximum_filter(distance, size=5)) & (distance > min_distance)
    markers, num_markers = scipy_label(local_max)

    if num_markers <= 1:
        return mask

    # Use negative probability as the landscape for watershed
    # (watershed finds basins, so we invert)
    landscape = -pr * mask

    # Apply watershed
    labels = watershed(landscape, markers, mask=mask)

    # Convert back to binary, removing touching regions
    # Dilate each label slightly and check for overlaps
    result = np.zeros_like(mask)
    for i in range(1, num_markers + 1):
        component = (labels == i).astype(np.uint8)
        result = np.maximum(result, component)

    return result


def skeleton_reconstruction(
    mask: np.ndarray,
    ribbon_width: int = 3,
) -> np.ndarray:
    """Reconstruct ribbons from skeleton to ensure consistent width.

    Skeletonizes the mask then dilates to achieve target width.
    Helps regularize ribbon width and remove protrusions.
    """
    if mask.sum() == 0:
        return mask

    # Skeletonize
    skeleton = skeletonize_3d(mask.astype(bool))

    # Dilate skeleton to target width
    # Use anisotropic structuring element (thinner in z)
    struct = np.zeros((3, ribbon_width, ribbon_width), dtype=bool)
    struct[1] = disk(ribbon_width // 2)  # Disk in xy plane
    struct[0] = struct[2] = disk(max(1, ribbon_width // 2 - 1))  # Smaller in z

    result = binary_dilation(skeleton, structure=struct).astype(np.uint8)
    return result


def morphological_cleanup(
    mask: np.ndarray,
    closing_size: int = 2,
    opening_size: int = 1,
) -> np.ndarray:
    """Apply morphological operations to clean up the mask.

    Closing fills small gaps, opening removes small protrusions.
    Uses anisotropic structuring elements suited for z-extending ribbons.
    """
    # Anisotropic structuring element (larger in xy, smaller in z)
    struct_close = np.zeros((closing_size, closing_size * 2 + 1, closing_size * 2 + 1), dtype=bool)
    for z in range(closing_size):
        struct_close[z] = disk(closing_size)

    struct_open = np.zeros((opening_size, opening_size * 2 + 1, opening_size * 2 + 1), dtype=bool)
    for z in range(opening_size):
        struct_open[z] = disk(opening_size)

    # Apply closing then opening (use skimage.morphology.closing/opening)
    result = morph_closing(mask.astype(bool), struct_close)
    result = morph_opening(result, struct_open)

    return result.astype(np.uint8)


def probability_guided_refinement(
    mask: np.ndarray,
    pr: np.ndarray,
    grow_threshold: float = 0.3,
    shrink_threshold: float = 0.1,
    iterations: int = 2,
) -> np.ndarray:
    """Refine mask boundaries using probability values.

    Iteratively grows into high-probability regions and shrinks from
    low-probability regions to better align with the model's predictions.
    """
    result = mask.copy()

    for _ in range(iterations):
        # Grow: dilate and keep pixels above grow_threshold
        dilated = binary_dilation(result)
        grow_mask = dilated & (pr > grow_threshold)

        # Shrink: erode and add back pixels above shrink_threshold
        eroded = binary_erosion(result)
        shrink_mask = result & (pr < shrink_threshold) & ~eroded

        result = (result | grow_mask) & ~shrink_mask

    return result.astype(np.uint8)


def optimal_threshold_search(
    pr: np.ndarray,
    thresholds: list = None,
    metric: str = "connectivity",
) -> tuple[float, np.ndarray]:
    """Search for optimal threshold based on structural metrics.

    Since we can't use ground truth, we use structural priors:
    - connectivity: prefer thresholds that give well-connected components
    - z_span: prefer thresholds where components span more z-slices
    - border: prefer thresholds where components reach borders
    """
    if thresholds is None:
        thresholds = np.arange(0.1, 0.5, 0.05)

    best_score = -1
    best_threshold = 0.2
    best_mask = None

    for thresh in thresholds:
        mask = (pr > thresh).astype(np.uint8)

        if mask.sum() == 0:
            continue

        labeled, num_features = scipy_label(mask)

        if num_features == 0:
            continue

        if metric == "connectivity":
            # Score based on average component size (larger = more connected)
            sizes = [np.sum(labeled == i) for i in range(1, num_features + 1)]
            score = np.mean(sizes) if sizes else 0

        elif metric == "z_span":
            # Score based on average z-span of components
            spans = []
            for i in range(1, num_features + 1):
                component = labeled == i
                z_indices = np.where(component.any(axis=(1, 2)))[0]
                if len(z_indices) > 0:
                    spans.append(z_indices[-1] - z_indices[0])
            score = np.mean(spans) if spans else 0

        elif metric == "border":
            # Score based on fraction of components reaching borders
            h, w = mask.shape[1], mask.shape[2]
            border_count = 0
            for i in range(1, num_features + 1):
                component = labeled == i
                xy_proj = component.any(axis=0)
                if (
                    xy_proj[0, :].any()
                    or xy_proj[-1, :].any()
                    or xy_proj[:, 0].any()
                    or xy_proj[:, -1].any()
                ):
                    border_count += 1
            score = border_count / num_features if num_features > 0 else 0

        if score > best_score:
            best_score = score
            best_threshold = thresh
            best_mask = mask

    return best_threshold, best_mask


def ensemble_thresholds(
    pr: np.ndarray,
    thresholds: list = [0.1, 0.15, 0.2, 0.25, 0.3],
    voting: str = "majority",
) -> np.ndarray:
    """Combine multiple threshold results through voting.

    More robust than single threshold as it reduces sensitivity to
    threshold choice.
    """
    masks = np.stack([(pr > t).astype(np.uint8) for t in thresholds], axis=0)

    if voting == "majority":
        # Pixel is foreground if majority of thresholds agree
        result = (masks.sum(axis=0) > len(thresholds) // 2).astype(np.uint8)
    elif voting == "any":
        # Pixel is foreground if any threshold includes it
        result = (masks.sum(axis=0) > 0).astype(np.uint8)
    elif voting == "all":
        # Pixel is foreground if all thresholds include it
        result = (masks.sum(axis=0) == len(thresholds)).astype(np.uint8)
    elif voting == "weighted":
        # Weight by threshold (higher threshold = more confident)
        weights = np.array(thresholds).reshape(-1, 1, 1, 1)
        weighted_sum = (masks * weights).sum(axis=0)
        result = (weighted_sum > sum(thresholds) / 2).astype(np.uint8)

    return result


def create_postprocess_pipeline(
    steps: list[str],
    params: dict = None,
) -> Callable:
    """Create a post-processing pipeline from a list of step names.

    Available steps:
    - 'threshold': simple_threshold
    - 'hysteresis': hysteresis_threshold
    - 'adaptive': adaptive_threshold_3d
    - 'fill_holes_2d': fill_holes_2d_per_slice
    - 'fill_holes_3d': fill_holes_3d
    - 'remove_small': remove_small_components
    - 'z_continuity': enforce_z_continuity
    - 'border': enforce_border_connectivity
    - 'separate': separate_touching_ribbons
    - 'skeleton': skeleton_reconstruction
    - 'morphology': morphological_cleanup
    - 'refine': probability_guided_refinement
    - 'ensemble': ensemble_thresholds
    """
    if params is None:
        params = {}

    step_funcs = {
        "threshold": simple_threshold,
        "hysteresis": hysteresis_threshold,
        "adaptive": adaptive_threshold_3d,
        "surfaceness": surfaceness_threshold,
        "fill_holes_2d": fill_holes_2d_per_slice,
        "fill_holes_3d": fill_holes_3d,
        "remove_small": remove_small_components,
        "z_continuity": enforce_z_continuity,
        "border": enforce_border_connectivity,
        "separate": separate_touching_ribbons,
        "skeleton": skeleton_reconstruction,
        "morphology": morphological_cleanup,
        "refine": probability_guided_refinement,
        "ensemble": ensemble_thresholds,
    }

    def pipeline(pr: np.ndarray) -> np.ndarray:
        result = pr
        mask = None

        for step in steps:
            func = step_funcs[step]
            step_params = params.get(step, {})

            # Some functions need the probability map, others work on mask
            if step in ["threshold", "hysteresis", "adaptive", "ensemble", "surfaceness"]:
                mask = func(pr, **step_params)
            elif step in ["separate", "refine"]:
                mask = func(mask, pr, **step_params)
            else:
                mask = func(mask, **step_params)

        return mask

    return pipeline


# Predefined pipelines optimized for thin ribbon segmentation
# Updated based on GT analysis of 200 volumes
PIPELINES = {
    "baseline": {
        "steps": ["threshold"],
        "params": {"threshold": {"threshold": 0.2}},
    },
    # Conservative: safe improvements, minimal risk of removing real ribbons
    "conservative": {
        "steps": ["hysteresis", "fill_holes_2d", "remove_small"],
        "params": {
            "hysteresis": {"low_threshold": 0.1, "high_threshold": 0.3},
            "remove_small": {"min_size": 50},  # Keep small to avoid losing fragments
        },
    },
    # GT-optimized: based on ground truth analysis
    "gt_optimized": {
        "steps": ["hysteresis", "fill_holes_2d", "fill_holes_3d", "remove_small", "z_continuity"],
        "params": {
            "hysteresis": {"low_threshold": 0.08, "high_threshold": 0.25},
            "fill_holes_3d": {"max_hole_size": 100},
            "remove_small": {"min_size": 100},  # GT 5th percentile for real ribbons
            "z_continuity": {"min_z_span": 10, "min_overlap_ratio": 0.2},  # Based on GT overlap
        },
    },
    # Aggressive: filters more aggressively, may lose some small valid regions
    "aggressive_cleanup": {
        "steps": [
            "threshold",
            "morphology",
            "fill_holes_3d",
            "remove_small",
            "z_continuity",
        ],
        "params": {
            "threshold": {"threshold": 0.15},
            "morphology": {"closing_size": 2, "opening_size": 1},
            "fill_holes_3d": {"max_hole_size": 100},
            "remove_small": {"min_size": 500},  # Aggressive size filter
            "z_continuity": {"min_z_span": 20},
        },
    },
    # Topology-focused: optimizes for topological correctness
    "topology_focused": {
        "steps": [
            "hysteresis",
            "fill_holes_2d",
            "fill_holes_3d",
            "morphology",
            "remove_small",
            "z_continuity",
        ],
        "params": {
            "hysteresis": {"low_threshold": 0.10, "high_threshold": 0.28},
            "fill_holes_3d": {"max_hole_size": 150},
            "morphology": {"closing_size": 2, "opening_size": 1},
            "remove_small": {"min_size": 200},
            "z_continuity": {"min_z_span": 15, "min_overlap_ratio": 0.15},
        },
    },
    # Ensemble-robust: stable across threshold variations
    "ensemble_robust": {
        "steps": [
            "ensemble",
            "fill_holes_2d",
            "fill_holes_3d",
            "morphology",
            "remove_small",
            "z_continuity",
        ],
        "params": {
            "ensemble": {"thresholds": [0.1, 0.15, 0.2, 0.25], "voting": "majority"},
            "fill_holes_3d": {"max_hole_size": 100},
            "morphology": {"closing_size": 2},
            "remove_small": {"min_size": 150},
            "z_continuity": {"min_z_span": 10},
        },
    },
    # Skeleton-based: reconstructs ribbons with controlled width (~8-9 voxels from GT)
    "skeleton_based": {
        "steps": ["threshold", "skeleton", "fill_holes_2d", "remove_small"],
        "params": {
            "threshold": {"threshold": 0.15},
            "skeleton": {"ribbon_width": 4},  # radius=4 gives ~8 voxel diameter
            "remove_small": {"min_size": 100},
        },
    },
    # Minimal: just hole filling and small component removal
    "minimal": {
        "steps": ["threshold", "fill_holes_2d", "remove_small"],
        "params": {
            "threshold": {"threshold": 0.18},
            "remove_small": {"min_size": 50},
        },
    },
    # VOI-optimized: focus on reducing merge/split errors (without separate step)
    "voi_optimized": {
        "steps": ["threshold", "fill_holes_2d", "remove_small"],
        "params": {
            "threshold": {"threshold": 0.22},
            "remove_small": {"min_size": 30},  # Very conservative
        },
    },
    # Balanced: slight improvements without hurting surface dice
    "balanced": {
        "steps": ["threshold", "fill_holes_2d", "remove_small"],
        "params": {
            "threshold": {"threshold": 0.20},
            "remove_small": {"min_size": 25},  # Very minimal cleanup
        },
    },
    # Surface-preserving: focus on not changing boundaries
    "surface_preserving": {
        "steps": ["threshold", "fill_holes_2d"],
        "params": {
            "threshold": {"threshold": 0.20},
        },
    },
    # Topo-tuned: focus on topology with less aggressive filtering
    "topo_tuned": {
        "steps": ["hysteresis", "fill_holes_2d", "remove_small"],
        "params": {
            "hysteresis": {"low_threshold": 0.15, "high_threshold": 0.30},
            "remove_small": {"min_size": 30},
        },
    },
    # Surfaceness-enhanced: uses modified Frangi filter for sheet-like structures
    # Thresholds at 0.75 and applies surfaceness filter to enhance ribbons
    "surfaceness_enhanced": {
        "steps": ["surfaceness"],
        "params": {
            "surfaceness": {
                "threshold": 0.75,
                "sigmas": [1.0, 2.0],
                "alpha": 0.5,
                "beta": 0.5,
                "combine_method": "enhance",
            },
        },
    },
}


def apply_pipeline(
    pr: np.ndarray,
    pipeline_name: str = "conservative",
    custom_params: dict = None,
) -> np.ndarray:
    """Apply a predefined post-processing pipeline.

    Args:
        pr: Probability map (float32, 0-1 range)
        pipeline_name: Name of predefined pipeline or 'custom'
        custom_params: Override parameters for the pipeline

    Returns:
        Binary mask (uint8)

    """
    if pipeline_name not in PIPELINES:
        raise ValueError(f"Unknown pipeline: {pipeline_name}. Available: {list(PIPELINES.keys())}")

    config = PIPELINES[pipeline_name].copy()

    if custom_params:
        for step, params in custom_params.items():
            if step in config["params"]:
                config["params"][step].update(params)
            else:
                config["params"][step] = params

    pipeline = create_postprocess_pipeline(config["steps"], config["params"])
    return pipeline(pr)


if __name__ == "__main__":
    # Example usage and testing
    print("Available pipelines:")
    for name, config in PIPELINES.items():
        print(f"  {name}: {' -> '.join(config['steps'])}")

    # Create synthetic test data
    print("\nCreating synthetic ribbon data for testing...")
    shape = (50, 64, 64)

    # Create a synthetic ribbon (diagonal in xy, full z-span)
    pr = np.zeros(shape, dtype=np.float32)
    for z in range(shape[0]):
        for i in range(-2, 3):
            y = np.clip(np.arange(shape[1]) + i, 0, shape[1] - 1)
            x = np.clip(np.arange(shape[2]) + i + z // 5, 0, shape[2] - 1)
            pr[z, y, x] = 0.8 - 0.1 * abs(i)

    # Add some noise
    pr += np.random.randn(*shape).astype(np.float32) * 0.1
    pr = np.clip(pr, 0, 1)

    print(f"Test data shape: {pr.shape}")
    print(f"Probability range: [{pr.min():.3f}, {pr.max():.3f}]")

    # Test each pipeline
    for name in PIPELINES:
        mask = apply_pipeline(pr, name)
        labeled, num = scipy_label(mask)
        print(f"\n{name}:")
        print(f"  Foreground voxels: {mask.sum()}")
        print(f"  Connected components: {num}")
