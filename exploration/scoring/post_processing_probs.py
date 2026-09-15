# --- Advanced Topology-Aware Postprocessing with cc3d ---
import cc3d
import numpy as np
from scipy.ndimage import binary_closing, binary_fill_holes, generate_binary_structure
from skimage.morphology import skeletonize


def postprocess_cc3d_topology(
    pr: np.ndarray,
    threshold: float = 0.18,
    min_size: int = 500,
    closing_radius: int = 2,
    fill_holes: bool = True,
    skeleton_gap_bridge: bool = False,
    shape_filter: bool = False,
    compactness_thresh: float = 0.1,
    **kwargs,
) -> np.ndarray:
    """Advanced topology-aware postprocessing using cc3d and 3D morphology.
    Steps:
      1. Threshold
      2. 3D morphological closing
      3. Connected components (cc3d)
      4. Remove small components
      5. (Optional) Shape filtering
      6. (Optional) Skeleton-based gap bridging
      7. 3D hole filling
    """
    # 1. Threshold
    mask = (pr > threshold).astype(np.uint8)

    # 2. 3D morphological closing to bridge small gaps
    if closing_radius > 0:
        struct = generate_binary_structure(3, 2)
        mask = binary_closing(mask, structure=struct, iterations=closing_radius)

    # 3. Connected components labeling
    labels, n_labels = cc3d.connected_components(mask, connectivity=26, return_N=True)

    # 4. Remove small components using cc3d.dust()
    labels = cc3d.dust(labels, threshold=min_size, connectivity=26, in_place=False)

    # 5. (Optional) Shape filtering (by z-span and elongation)
    if shape_filter:
        stats = cc3d.statistics(labels)
        keep_labels = []
        for label_id in range(1, n_labels + 1):
            if label_id >= len(stats["voxel_counts"]):
                continue
            voxel_count = stats["voxel_counts"][label_id]
            if voxel_count < min_size:
                continue
            # Get bounding box for shape analysis
            bbox = stats["bounding_boxes"][label_id]
            z_span = bbox[0].stop - bbox[0].start
            y_span = bbox[1].stop - bbox[1].start
            x_span = bbox[2].stop - bbox[2].start
            # Keep components with good z-span and elongation
            if z_span >= 20:  # Minimum z-span
                keep_labels.append(label_id)
        mask = np.isin(labels, keep_labels).astype(np.uint8)
    else:
        mask = (labels > 0).astype(np.uint8)

    # 6. (Optional) Skeleton-based gap bridging (not implemented, placeholder)
    # Could be added: skeletonize_3d, find endpoints, bridge small gaps

    # 7. 3D hole filling
    if fill_holes:
        mask = binary_fill_holes(mask).astype(np.uint8)

    return mask


# =============================================================================
# ADVANCED CC3D-BASED TOPOLOGY PROCESSING (NEW)
# =============================================================================
# These methods leverage cc3d's advanced features:
# - cc3d.dust() for size filtering with threshold ranges
# - cc3d.largest_k() for keeping top-K components
# - cc3d.statistics() for shape-based filtering
# - cc3d.contacts() for detecting false connections
# - cc3d.region_graph() for component adjacency analysis
# =============================================================================


def cc3d_dust_filter(
    mask: np.ndarray,
    min_size: int = 500,
    max_size: int = None,
    connectivity: int = 26,
) -> np.ndarray:
    """Remove dust (small components) using cc3d.dust().

    cc3d.dust() is more efficient than remove_small_objects for this task.
    Can also remove very large components if max_size is specified.

    Args:
        mask: Binary mask
        min_size: Minimum component size to keep
        max_size: Maximum component size to keep (None = no limit)
        connectivity: 6, 18, or 26

    Returns:
        Filtered binary mask

    """
    labels = cc3d.connected_components(mask.astype(np.uint8), connectivity=connectivity)

    if max_size is not None:
        # Remove both small and large components
        labels = cc3d.dust(labels, threshold=[min_size, max_size], connectivity=connectivity)
    else:
        # Remove only small components
        labels = cc3d.dust(labels, threshold=min_size, connectivity=connectivity)

    return (labels > 0).astype(np.uint8)


def cc3d_largest_k(
    mask: np.ndarray,
    k: int = 10,
    connectivity: int = 26,
) -> np.ndarray:
    """Keep only the k largest connected components.

    Very effective for removing noise when you know the approximate
    number of real objects.

    Args:
        mask: Binary mask
        k: Number of largest components to keep
        connectivity: 6, 18, or 26

    Returns:
        Filtered binary mask with only k largest components

    """
    labels, n = cc3d.largest_k(
        mask.astype(np.uint8),
        k=k,
        connectivity=connectivity,
        return_N=True,
    )
    return (labels > 0).astype(np.uint8)


def cc3d_shape_filter(
    mask: np.ndarray,
    min_size: int = 500,
    min_z_span: int = 30,
    max_compactness: float = 50.0,
    min_elongation: float = 2.0,
    connectivity: int = 26,
) -> np.ndarray:
    """Filter components by shape characteristics using cc3d.statistics().

    Ribbons have characteristic shape signatures:
    - High elongation (thin and long)
    - Large z-span (extend through multiple slices)

    Args:
        mask: Binary mask
        min_size: Minimum voxel count
        min_z_span: Minimum z-extent
        max_compactness: Maximum compactness (not used, kept for compatibility)
        min_elongation: Minimum elongation ratio
        connectivity: 6, 18, or 26

    Returns:
        Shape-filtered binary mask

    """
    labels, n_labels = cc3d.connected_components(
        mask.astype(np.uint8),
        connectivity=connectivity,
        return_N=True,
    )

    if n_labels == 0:
        return mask

    stats = cc3d.statistics(labels)
    keep_labels = []

    for label_id in range(1, n_labels + 1):
        if label_id >= len(stats["voxel_counts"]):
            continue

        voxel_count = stats["voxel_counts"][label_id]

        # Skip small components
        if voxel_count < min_size:
            continue

        # Get bounding box - format is slice objects (slice(start, stop), ...)
        bbox = stats["bounding_boxes"][label_id]
        z_extent = bbox[0].stop - bbox[0].start
        y_extent = bbox[1].stop - bbox[1].start
        x_extent = bbox[2].stop - bbox[2].start

        # Skip components with insufficient z-span
        if z_extent < min_z_span:
            continue

        # Elongation: ratio of largest to smallest dimension
        extents = sorted([z_extent, y_extent, x_extent])
        if extents[0] > 0:
            elongation = extents[2] / (extents[0] + 1e-6)
        else:
            elongation = float("inf")

        # Apply filters
        if elongation >= min_elongation:
            keep_labels.append(label_id)

    return np.isin(labels, keep_labels).astype(np.uint8)


def cc3d_contact_analysis(
    mask: np.ndarray,
    max_contact_area: int = 50,
    min_size_for_merge: int = 1000,
    connectivity: int = 26,
) -> np.ndarray:
    """Analyze and potentially separate components with suspicious contacts.

    GT analysis shows ribbons NEVER touch (0% contact rate).
    If two components have significant contact, they might be falsely merged.

    This function:
    1. Labels components
    2. Computes contact surface areas between all pairs
    3. Components with large contact areas are flagged
    4. Small contacts are treated as noise and can be eroded away

    Args:
        mask: Binary mask
        max_contact_area: Maximum contact area before flagging as suspicious
        min_size_for_merge: Minimum size to consider for contact analysis
        connectivity: 6, 18, or 26

    Returns:
        Mask with false connections potentially removed

    """
    labels, n_labels = cc3d.connected_components(
        mask.astype(np.uint8),
        connectivity=connectivity,
        return_N=True,
    )

    if n_labels < 2:
        return mask

    # Get contact surface areas between components
    contacts = cc3d.contacts(labels, connectivity=6, surface_area=True)

    # Find pairs with suspiciously high contact (potential false merges)
    # For now, we just flag them - actual separation would require watershed
    suspicious_pairs = []
    for (l1, l2), contact_area in contacts.items():
        if l1 == 0 or l2 == 0:  # Skip background contacts
            continue
        if contact_area > max_contact_area:
            suspicious_pairs.append((l1, l2, contact_area))

    # For components with suspicious contacts, apply light erosion
    # to potentially separate them
    if suspicious_pairs:
        from scipy.ndimage import binary_erosion

        struct = generate_binary_structure(3, 1)  # 6-connectivity erosion
        eroded = binary_erosion(mask, structure=struct, iterations=1)
        # Re-dilate to recover size but maintain separation
        from scipy.ndimage import binary_dilation

        result = binary_dilation(eroded, structure=struct, iterations=1)
        return result.astype(np.uint8)

    return mask


def cc3d_region_graph_analysis(
    mask: np.ndarray,
    connectivity: int = 26,
) -> tuple[np.ndarray, dict]:
    """Extract region graph for advanced topology analysis.

    Returns the mask and a graph showing which components neighbor each other.
    This can be used for downstream analysis like:
    - Detecting ribbon bundles
    - Finding isolated components
    - Graph-based cleaning

    Args:
        mask: Binary mask
        connectivity: 6, 18, or 26

    Returns:
        Tuple of (labels, edges) where edges is a set of (label1, label2) pairs

    """
    labels = cc3d.connected_components(mask.astype(np.uint8), connectivity=connectivity)
    edges = cc3d.region_graph(labels, connectivity=connectivity)
    return labels, edges


def cc3d_continuous_threshold(
    pr: np.ndarray,
    delta: float = 0.05,
    min_size: int = 500,
    connectivity: int = 26,
) -> np.ndarray:
    """Use cc3d's continuous value CCL for soft thresholding.

    Instead of hard thresholding, this uses cc3d's delta parameter
    to group nearby probability values into the same component.
    This can help bridge small gaps in probability maps.

    Args:
        pr: Probability map (0-1)
        delta: Maximum probability difference to consider same component
        min_size: Minimum component size
        connectivity: 6, 18, or 26

    Returns:
        Binary mask

    """
    # Scale probabilities to integer range for cc3d
    # cc3d's delta works on the actual values
    pr_scaled = (pr * 255).astype(np.uint8)
    delta_scaled = int(delta * 255)

    # Use continuous value CCL
    labels = cc3d.connected_components(
        pr_scaled,
        connectivity=connectivity,
        delta=delta_scaled,
    )

    # Remove small components
    labels = cc3d.dust(labels, threshold=min_size, connectivity=connectivity)

    return (labels > 0).astype(np.uint8)


def cc3d_iterative_refinement(
    pr: np.ndarray,
    thresholds: list[float] = [0.3, 0.2, 0.15, 0.1],
    min_sizes: list[int] = [2000, 1000, 500, 300],
    connectivity: int = 26,
) -> np.ndarray:
    """Iterative component refinement with progressively lower thresholds.

    Start with high threshold to get confident seeds, then progressively
    lower threshold while using previous components as constraints.

    This is similar to hysteresis but with multiple levels.

    Args:
        pr: Probability map
        thresholds: List of thresholds from high to low
        min_sizes: Corresponding minimum sizes for each threshold
        connectivity: 6, 18, or 26

    Returns:
        Refined binary mask

    """
    assert len(thresholds) == len(min_sizes), "thresholds and min_sizes must match"

    # Start with highest threshold
    mask = (pr > thresholds[0]).astype(np.uint8)
    labels, _ = cc3d.connected_components(mask, connectivity=connectivity, return_N=True)
    labels = cc3d.dust(labels, threshold=min_sizes[0], connectivity=connectivity)

    # Iteratively expand with lower thresholds
    for thresh, min_size in zip(thresholds[1:], min_sizes[1:]):
        # Get mask at current threshold
        current_mask = (pr > thresh).astype(np.uint8)

        # Label the new mask
        current_labels, _ = cc3d.connected_components(
            current_mask,
            connectivity=connectivity,
            return_N=True,
        )

        # Keep components that overlap with existing labels
        overlap_labels = np.unique(current_labels[labels > 0])
        overlap_labels = overlap_labels[overlap_labels > 0]

        # Expand labels to include overlapping regions
        expanded = np.isin(current_labels, overlap_labels)
        labels = cc3d.connected_components(
            expanded.astype(np.uint8),
            connectivity=connectivity,
        )
        labels = cc3d.dust(labels, threshold=min_size, connectivity=connectivity)

    return (labels > 0).astype(np.uint8)


def cc3d_skeleton_bridge_gaps(
    mask: np.ndarray,
    max_gap: int = 5,
    min_component_size: int = 500,
    connectivity: int = 26,
) -> np.ndarray:
    """Bridge small gaps using skeletonization and dilation.

    Steps:
    1. Skeletonize each component
    2. Dilate skeleton to find potential bridge points
    3. Connect nearby endpoints
    4. Reconstruct using distance transform

    Args:
        mask: Binary mask
        max_gap: Maximum gap size to bridge
        min_component_size: Minimum component size to process
        connectivity: 6, 18, or 26

    Returns:
        Mask with gaps bridged

    """
    from scipy.ndimage import binary_dilation, distance_transform_edt

    labels, n_labels = cc3d.connected_components(
        mask.astype(np.uint8),
        connectivity=connectivity,
        return_N=True,
    )

    if n_labels == 0:
        return mask

    # Compute distance transform of background
    dist = distance_transform_edt(~mask.astype(bool))

    # Dilate mask to find potential bridge regions
    struct = generate_binary_structure(3, 2)
    dilated = binary_dilation(mask, structure=struct, iterations=max_gap)

    # Find new regions that would connect existing components
    bridge_candidates = dilated & ~mask.astype(bool) & (dist <= max_gap)

    # Check if bridges connect same or different components
    if bridge_candidates.any():
        # Label bridge candidates
        bridge_labels = cc3d.connected_components(
            bridge_candidates.astype(np.uint8),
            connectivity=connectivity,
        )

        # For each bridge region, check if it connects components
        for bridge_id in np.unique(bridge_labels):
            if bridge_id == 0:
                continue

            bridge_mask = bridge_labels == bridge_id
            # Dilate bridge slightly to find adjacent components
            bridge_dilated = binary_dilation(bridge_mask, structure=struct)
            adjacent_labels = np.unique(labels[bridge_dilated & (labels > 0)])

            # Only bridge if it connects exactly 2 different components
            if len(adjacent_labels) == 2:
                # Check sizes of components being connected
                sizes = [np.sum(labels == l) for l in adjacent_labels]
                if all(s >= min_component_size for s in sizes):
                    # Add bridge to mask
                    mask = mask | bridge_mask.astype(np.uint8)

    return mask


def cc3d_watershed_separation(
    pr: np.ndarray,
    threshold: float = 0.18,
    seed_threshold: float = 0.35,
    min_size: int = 500,
) -> np.ndarray:
    """Use watershed to separate touching components.

    GT analysis shows ribbons never touch. If the thresholded mask
    has components that are connected, try to separate them using
    watershed with high-probability seeds.

    Args:
        pr: Probability map
        threshold: Threshold for watershed basin
        seed_threshold: Higher threshold for watershed seeds
        min_size: Minimum component size

    Returns:
        Separated binary mask

    """
    from scipy.ndimage import distance_transform_edt
    from skimage.segmentation import watershed

    # Create mask at lower threshold
    mask = (pr > threshold).astype(np.uint8)

    # Create seeds at higher threshold
    seeds = (pr > seed_threshold).astype(np.uint8)
    seed_labels = cc3d.connected_components(seeds, connectivity=26)

    # Distance transform for watershed
    dist = distance_transform_edt(mask)

    # Apply watershed
    ws_labels = watershed(-dist, seed_labels, mask=mask)

    # Remove small components
    ws_labels = cc3d.dust(ws_labels, threshold=min_size, connectivity=26)

    return (ws_labels > 0).astype(np.uint8)


def cc3d_per_component_processing(
    mask: np.ndarray,
    fill_holes: bool = True,
    smooth: bool = True,
    min_size: int = 500,
    connectivity: int = 26,
) -> np.ndarray:
    """Process each connected component individually.

    Using cc3d.each() for efficient iteration over components.
    Apply hole filling and smoothing per-component to avoid
    accidentally merging separate ribbons.

    Args:
        mask: Binary mask
        fill_holes: Fill holes in each component
        smooth: Apply morphological smoothing
        min_size: Minimum component size
        connectivity: 6, 18, or 26

    Returns:
        Processed binary mask

    """
    labels = cc3d.connected_components(mask.astype(np.uint8), connectivity=connectivity)
    labels = cc3d.dust(labels, threshold=min_size, connectivity=connectivity)

    result = np.zeros_like(mask)

    # Process each component efficiently
    for label_id, component_mask in cc3d.each(labels, binary=True, in_place=True):
        if label_id == 0:
            continue

        processed = component_mask.copy()

        if fill_holes:
            # Fill 2D holes slice by slice
            for z in range(processed.shape[0]):
                processed[z] = binary_fill_holes(processed[z])

        if smooth:
            # Light morphological smoothing
            struct = generate_binary_structure(3, 1)
            processed = binary_closing(processed, structure=struct, iterations=1)

        result = np.maximum(result, processed.astype(np.uint8))

    return result


def cc3d_statistics_filter(
    mask: np.ndarray,
    min_size: int = 500,
    min_z_span_ratio: float = 0.1,
    max_surface_volume_ratio: float = 5.0,
    connectivity: int = 26,
) -> np.ndarray:
    """Advanced filtering using cc3d.statistics() metrics.

    Uses bounding box, voxel count, and centroid information
    for sophisticated shape-based filtering.

    Args:
        mask: Binary mask
        min_size: Minimum voxel count
        min_z_span_ratio: Minimum z-span as ratio of total z-depth
        max_surface_volume_ratio: Maximum surface-to-volume ratio
        connectivity: 6, 18, or 26

    Returns:
        Filtered binary mask

    """
    labels, n_labels = cc3d.connected_components(
        mask.astype(np.uint8),
        connectivity=connectivity,
        return_N=True,
    )

    if n_labels == 0:
        return mask

    stats = cc3d.statistics(labels)
    z_depth = mask.shape[0]
    keep_labels = []

    for label_id in range(1, n_labels + 1):
        if label_id >= len(stats["voxel_counts"]):
            continue

        voxel_count = stats["voxel_counts"][label_id]
        bbox = stats["bounding_boxes"][label_id]

        # Size filter
        if voxel_count < min_size:
            continue

        # Z-span filter - bbox is slice objects
        z_span = bbox[0].stop - bbox[0].start
        if z_span / z_depth < min_z_span_ratio:
            continue

        # Surface-to-volume ratio (approximate from bbox)
        z_ext = bbox[0].stop - bbox[0].start
        y_ext = bbox[1].stop - bbox[1].start
        x_ext = bbox[2].stop - bbox[2].start
        bbox_surface = 2 * (z_ext * y_ext + y_ext * x_ext + z_ext * x_ext)
        bbox_volume = z_ext * y_ext * x_ext

        if bbox_volume > 0:
            sv_ratio = bbox_surface / bbox_volume
            if sv_ratio > max_surface_volume_ratio:
                continue

        keep_labels.append(label_id)

    return np.isin(labels, keep_labels).astype(np.uint8)


# =============================================================================
# COMPREHENSIVE CC3D PIPELINES
# =============================================================================


def postprocess_cc3d_advanced(
    pr: np.ndarray,
    threshold: float = 0.18,
    min_size: int = 500,
    use_shape_filter: bool = True,
    use_contact_analysis: bool = True,
    use_hole_filling: bool = True,
    closing_radius: int = 1,
    connectivity: int = 26,
    **kwargs,
) -> np.ndarray:
    """Advanced cc3d-based pipeline combining multiple techniques.

    Steps:
    1. Threshold
    2. Morphological closing to bridge small gaps
    3. cc3d connected components
    4. Dust removal (small component filtering)
    5. Shape-based filtering
    6. Contact analysis (detect false connections)
    7. Per-component hole filling

    Args:
        pr: Probability map
        threshold: Probability threshold
        min_size: Minimum component size
        use_shape_filter: Apply shape-based filtering
        use_contact_analysis: Analyze and handle contacts
        use_hole_filling: Fill holes in components
        closing_radius: Morphological closing radius
        connectivity: 6, 18, or 26

    Returns:
        Processed binary mask

    """
    # 1. Threshold
    mask = (pr > threshold).astype(np.uint8)

    # 2. Morphological closing
    if closing_radius > 0:
        struct = generate_binary_structure(3, 2)
        mask = binary_closing(mask, structure=struct, iterations=closing_radius)

    # 3-4. CC3D labeling and dust removal
    labels = cc3d.connected_components(mask, connectivity=connectivity)
    labels = cc3d.dust(labels, threshold=min_size, connectivity=connectivity)
    mask = (labels > 0).astype(np.uint8)

    # 5. Shape filtering
    if use_shape_filter:
        mask = cc3d_shape_filter(
            mask,
            min_size=min_size,
            min_z_span=30,
            connectivity=connectivity,
        )

    # 6. Contact analysis
    if use_contact_analysis:
        mask = cc3d_contact_analysis(
            mask,
            max_contact_area=50,
            connectivity=connectivity,
        )

    # 7. Per-component hole filling
    if use_hole_filling:
        mask = cc3d_per_component_processing(
            mask,
            fill_holes=True,
            smooth=False,
            min_size=min_size,
            connectivity=connectivity,
        )

    return mask


def postprocess_cc3d_iterative(
    pr: np.ndarray,
    high_threshold: float = 0.30,
    low_threshold: float = 0.10,
    num_levels: int = 4,
    min_size: int = 500,
    connectivity: int = 26,
    **kwargs,
) -> np.ndarray:
    """Iterative refinement pipeline using multiple threshold levels.

    Similar to hysteresis but with multiple intermediate levels for
    smoother expansion from high-confidence seeds.

    Args:
        pr: Probability map
        high_threshold: Starting (highest) threshold
        low_threshold: Ending (lowest) threshold
        num_levels: Number of threshold levels
        min_size: Minimum component size
        connectivity: 6, 18, or 26

    Returns:
        Refined binary mask

    """
    thresholds = np.linspace(high_threshold, low_threshold, num_levels).tolist()
    # Progressively relax size constraints
    min_sizes = [max(min_size, int(min_size * (1 + 0.5 * i))) for i in range(num_levels)]
    min_sizes.reverse()  # Larger sizes for higher thresholds

    return cc3d_iterative_refinement(
        pr,
        thresholds=thresholds,
        min_sizes=min_sizes,
        connectivity=connectivity,
    )


def postprocess_cc3d_topology_strict(
    pr: np.ndarray,
    low_threshold: float = 0.12,
    high_threshold: float = 0.28,
    min_size: int = 1000,
    min_z_span: int = 40,
    connectivity: int = 26,
    **kwargs,
) -> np.ndarray:
    """Strict topology-focused pipeline for high-quality segmentation.

    Prioritizes topological correctness over coverage.

    Args:
        pr: Probability map
        low_threshold: Low threshold for hysteresis
        high_threshold: High threshold for hysteresis seeds
        min_size: Minimum component size
        min_z_span: Minimum z-span
        connectivity: 6, 18, or 26

    Returns:
        Topologically clean binary mask

    """
    # 1. Hysteresis thresholding
    high_mask = pr > high_threshold
    low_mask = pr > low_threshold

    # Use cc3d for connected components
    labels = cc3d.connected_components(low_mask.astype(np.uint8), connectivity=connectivity)

    # Keep only components containing high-confidence voxels
    high_labels = np.unique(labels[high_mask])
    high_labels = high_labels[high_labels > 0]
    mask = np.isin(labels, high_labels).astype(np.uint8)

    # 2. Shape filtering with strict parameters
    mask = cc3d_shape_filter(
        mask,
        min_size=min_size,
        min_z_span=min_z_span,
        max_compactness=30.0,  # Stricter
        min_elongation=3.0,  # More elongated
        connectivity=connectivity,
    )

    # 3. Contact analysis to separate false merges
    mask = cc3d_contact_analysis(mask, max_contact_area=30, connectivity=connectivity)

    # 4. Per-component processing
    mask = cc3d_per_component_processing(
        mask,
        fill_holes=True,
        smooth=True,
        min_size=min_size,
        connectivity=connectivity,
    )

    return mask


def postprocess_cc3d_largest(
    pr: np.ndarray,
    threshold: float = 0.18,
    k: int = 20,
    min_size: int = 300,
    fill_holes: bool = True,
    connectivity: int = 26,
    **kwargs,
) -> np.ndarray:
    """Keep only the K largest components.

    Simple but effective when the number of real ribbons is known
    or can be estimated.

    Args:
        pr: Probability map
        threshold: Probability threshold
        k: Number of largest components to keep
        min_size: Additional minimum size filter
        fill_holes: Fill holes in final mask
        connectivity: 6, 18, or 26

    Returns:
        Binary mask with K largest components

    """
    mask = (pr > threshold).astype(np.uint8)

    # Keep K largest
    labels, n = cc3d.largest_k(mask, k=k, connectivity=connectivity, return_N=True)

    # Additional size filter
    labels = cc3d.dust(labels, threshold=min_size, connectivity=connectivity)

    mask = (labels > 0).astype(np.uint8)

    if fill_holes:
        mask = cc3d_per_component_processing(
            mask,
            fill_holes=True,
            smooth=False,
            min_size=0,  # Already filtered
            connectivity=connectivity,
        )

    return mask


def postprocess_cc3d_full(
    pr: np.ndarray,
    low_threshold: float = 0.10,
    high_threshold: float = 0.25,
    min_size: int = 800,
    min_z_span: int = 35,
    use_bridge_gaps: bool = True,
    use_watershed_separation: bool = False,
    use_statistics_filter: bool = True,
    connectivity: int = 26,
    **kwargs,
) -> np.ndarray:
    """Full cc3d-based pipeline with all features.

    Comprehensive pipeline that:
    1. Uses hysteresis thresholding
    2. Bridges small gaps
    3. Filters by statistics
    4. Separates touching components
    5. Fills holes per-component

    Args:
        pr: Probability map
        low_threshold: Low threshold for hysteresis
        high_threshold: High threshold for seeds
        min_size: Minimum component size
        min_z_span: Minimum z-span
        use_bridge_gaps: Bridge small gaps in ribbons
        use_watershed_separation: Use watershed to separate touching ribbons
        use_statistics_filter: Apply statistics-based filtering
        connectivity: 6, 18, or 26

    Returns:
        Fully processed binary mask

    """
    # 1. Hysteresis thresholding with cc3d
    high_mask = pr > high_threshold
    low_mask = pr > low_threshold

    labels = cc3d.connected_components(low_mask.astype(np.uint8), connectivity=connectivity)
    high_labels = np.unique(labels[high_mask])
    high_labels = high_labels[high_labels > 0]
    mask = np.isin(labels, high_labels).astype(np.uint8)

    # 2. Dust removal
    labels = cc3d.connected_components(mask, connectivity=connectivity)
    labels = cc3d.dust(labels, threshold=min_size // 2, connectivity=connectivity)
    mask = (labels > 0).astype(np.uint8)

    # 3. Bridge gaps
    if use_bridge_gaps:
        mask = cc3d_skeleton_bridge_gaps(
            mask,
            max_gap=3,
            min_component_size=min_size // 2,
            connectivity=connectivity,
        )

    # 4. Statistics-based filtering
    if use_statistics_filter:
        mask = cc3d_statistics_filter(
            mask,
            min_size=min_size,
            min_z_span_ratio=min_z_span / mask.shape[0],
            max_surface_volume_ratio=10.0,
            connectivity=connectivity,
        )

    # 5. Watershed separation (optional)
    if use_watershed_separation:
        mask = cc3d_watershed_separation(
            pr * mask,  # Mask the probability map
            threshold=low_threshold,
            seed_threshold=high_threshold,
            min_size=min_size,
        )

    # 6. Per-component hole filling
    mask = cc3d_per_component_processing(
        mask,
        fill_holes=True,
        smooth=True,
        min_size=min_size,
        connectivity=connectivity,
    )

    # 7. Final dust removal
    labels = cc3d.connected_components(mask, connectivity=connectivity)
    labels = cc3d.dust(labels, threshold=min_size, connectivity=connectivity)

    return (labels > 0).astype(np.uint8)


# Register new method in postprocess_gt
def postprocess_gt(pr, method="minimal", params=None):
    # ...existing code...
    if method == "cc3d_topology":
        return postprocess_cc3d_topology(pr, **params)
    # ...existing code...


"""GT-informed post-processing functions for 3D ribbon segmentation.

Based on comprehensive ground truth analysis of 786 volumes (6129 ribbons):

KEY GT FINDINGS:
===============
1. RIBBON WIDTH: 4.0 ± 0.3 voxels (range 3.6-4.2)
   - Use this for morphological structuring elements
   - Dilation radius for skeleton reconstruction: 2

2. Z-CONTINUITY: Mean z-span 278.6, 5th percentile 66.0
   - Ribbons are highly continuous along z-axis
   - Mean slice IoU overlap: 0.828
   - Safe min_z_span threshold: 46 voxels

3. SIZE FILTERING:
   - 1st percentile: 705 voxels
   - 5th percentile: 6100 voxels (safe minimum)
   - Tiny fragments (<352 voxels) are noise

4. HOLE CHARACTERISTICS:
   - Only 2.3% of GT ribbons have holes
   - Max hole size for filling: 100 voxels

5. TOPOLOGY:
   - Expected b0=1, b1≈0.04, b2≈0.01 per ribbon
   - Only 0.4% have tunnels, 0.0% have voids
   - Enforce no tunnels/voids in post-processing

6. INSTANCE SEPARATION:
   - Ribbons NEVER touch (0% in GT)
   - Min separation: 1.4 voxels
   - Mean separation: 70.2 voxels
   - Safe erosion radius: 1

7. BORDER CONNECTIVITY:
   - 0% of GT ribbons touch 2+ XY borders
   - DO NOT enforce border connectivity

8. MORPHOLOGY:
   - Closing size: 1 (based on ribbon width)
   - Opening size: 1

9. FRAGMENTATION:
   - GT fragmentation rate: 4.51%
   - Max gap to reconnect: 2 voxels

10. THICKNESS:
    - Uniformity: 83.9%
    - Max interior distance: 1.92 voxels
"""

import warnings
from collections.abc import Callable

import numpy as np
from scipy import ndimage
from scipy.ndimage import (
    binary_dilation,
    binary_erosion,
    binary_opening,
    distance_transform_edt,
    gaussian_filter,
    maximum_filter,
)
from scipy.ndimage import label as scipy_label
from skimage.measure import regionprops
from skimage.morphology import (
    ball,
    cube,
    disk,
    remove_small_holes,
    remove_small_objects,
)
from skimage.segmentation import watershed

warnings.filterwarnings("ignore", message=".*min_size.*deprecated.*")
warnings.filterwarnings("ignore", message=".*area_threshold.*deprecated.*")

# =============================================================================
# GT-INFORMED PARAMETERS (empirically tuned against ground-truth scoring)
# =============================================================================

GT_PARAMS = {
    "ribbon_width": 4.0,
    "ribbon_width_range": (3.6, 4.2),
    "dilation_radius": 2,
    "min_z_span": 46,
    "safe_min_z_span": 66,  # 5th percentile
    "mean_z_overlap": 0.828,
    "min_overlap_ratio": 0.23,
    "min_size_strict": 6100,  # 5th percentile
    "min_size_safe": 3050,  # More conservative
    "min_size_aggressive": 705,  # 1st percentile
    "max_hole_size": 100,
    "min_separation": 1.4,
    "safe_erosion_radius": 1,
    "closing_size": 1,
    "opening_size": 1,
    "fragmentation_gap": 2,
    "max_interior_distance": 1.92,
    "thickness_uniformity": 0.839,
    # NEW: Border connectivity parameters (from GT analysis)
    "border_start_rate": 0.996,  # 99.6% of ribbons start near border
    "border_end_rate": 0.996,  # 99.6% of ribbons end near border
    "border_standoff": 10,  # Distance from border to consider "near border"
    "path_straightness": 0.464,  # Expected path straightness
    "path_smoothness": 0.487,  # Expected path smoothness
    "max_centroid_displacement": 6.0,  # Max displacement per slice
    "mean_centroid_displacement": 2.0,  # Mean displacement per slice
}


# =============================================================================
# BASIC THRESHOLDING FUNCTIONS
# =============================================================================


def threshold_gt_optimized(
    pr: np.ndarray,
    threshold: float = 0.20,
) -> np.ndarray:
    """Simple threshold optimized based on GT ribbon characteristics.

    Args:
        pr: Probability map (0-1)
        threshold: Probability threshold

    Returns:
        Binary mask

    """
    return (pr > threshold).astype(np.uint8)


def hysteresis_gt_informed(
    pr: np.ndarray,
    low_threshold: float = 0.12,
    high_threshold: float = 0.28,
) -> np.ndarray:
    """Hysteresis thresholding with GT-informed gap.

    GT analysis suggests hysteresis_gap of 0.15 works well.
    Default gap: 0.28 - 0.12 = 0.16 (close to recommendation)

    Args:
        pr: Probability map
        low_threshold: Lower threshold for connectivity
        high_threshold: Higher threshold for seed regions

    Returns:
        Binary mask

    """
    high_mask = pr > high_threshold
    low_mask = pr > low_threshold

    # Label connected components in low threshold mask
    labeled, num_features = scipy_label(low_mask)

    # Keep components that contain high-confidence pixels
    high_labels = np.unique(labeled[high_mask])
    high_labels = high_labels[high_labels != 0]

    return np.isin(labeled, high_labels).astype(np.uint8)


# =============================================================================
# GT-INFORMED SIZE FILTERING
# =============================================================================


def remove_small_gt(
    mask: np.ndarray,
    min_size: int = None,
    mode: str = "safe",
) -> np.ndarray:
    """Remove small components based on GT size distribution.

    GT size percentiles:
    - 1st: 705 voxels
    - 5th: 6100 voxels
    - 10th: 27645 voxels

    Args:
        mask: Binary mask
        min_size: Override minimum size (if None, use mode)
        mode: "aggressive" (705), "safe" (3050), or "strict" (6100)

    Returns:
        Filtered binary mask

    """
    if min_size is None:
        min_size = {
            "aggressive": GT_PARAMS["min_size_aggressive"],
            "safe": GT_PARAMS["min_size_safe"],
            "strict": GT_PARAMS["min_size_strict"],
        }.get(mode, GT_PARAMS["min_size_safe"])

    return remove_small_objects(
        mask.astype(bool),
        min_size=min_size,
        connectivity=3,
    ).astype(np.uint8)


def remove_fragments_by_z_span(
    mask: np.ndarray,
    min_z_span: int = None,
) -> np.ndarray:
    """Remove components with insufficient z-span.

    GT z-span: mean=278.6, 5th percentile=66.0
    Default uses min_z_span=46 (conservative)

    Args:
        mask: Binary mask
        min_z_span: Minimum z-span required

    Returns:
        Filtered binary mask

    """
    if min_z_span is None:
        min_z_span = GT_PARAMS["min_z_span"]

    labeled, num_features = scipy_label(mask)
    result = np.zeros_like(mask)

    for i in range(1, num_features + 1):
        component = labeled == i
        z_indices = np.where(component.any(axis=(1, 2)))[0]

        if len(z_indices) >= min_z_span:
            result[component] = 1

    return result


# =============================================================================
# BORDER-AWARE FILTERING (KEY INSIGHT: 99.6% of ribbons start/end near border)
# =============================================================================


def _check_touches_border(slice_2d: np.ndarray, standoff: int = 10) -> dict:
    """Check if a 2D slice touches any XY border within standoff distance.

    Args:
        slice_2d: 2D binary mask (H, W)
        standoff: Distance from border to consider "touching"

    Returns:
        Dict with border touch info for each side

    """
    h, w = slice_2d.shape
    if slice_2d.sum() == 0:
        return {"top": False, "bottom": False, "left": False, "right": False, "any": False}

    # Check each border region
    top = slice_2d[:standoff, :].any()
    bottom = slice_2d[-standoff:, :].any()
    left = slice_2d[:, :standoff].any()
    right = slice_2d[:, -standoff:].any()

    return {
        "top": bool(top),
        "bottom": bool(bottom),
        "left": bool(left),
        "right": bool(right),
        "any": bool(top or bottom or left or right),
    }


def _get_component_endpoints(component: np.ndarray) -> tuple:
    """Get the start and end z-slices of a component.

    Returns:
        (start_slice, end_slice, z_indices) or (None, None, None) if empty

    """
    z_presence = component.any(axis=(1, 2))
    z_indices = np.where(z_presence)[0]

    if len(z_indices) == 0:
        return None, None, None

    return component[z_indices[0]], component[z_indices[-1]], z_indices


def filter_by_border_endpoints(
    mask: np.ndarray,
    border_standoff: int = None,
    require_start_near_border: bool = True,
    require_end_near_border: bool = True,
    min_endpoint_near_border: int = 1,
) -> np.ndarray:
    """Filter components to keep only those with endpoints near XY borders.

    KEY GT INSIGHT: 99.6% of ribbons start near border, 99.6% end near border.
    This is a very powerful filter for removing false positives that don't
    follow the expected ribbon structure.

    Args:
        mask: Binary mask (Z, H, W)
        border_standoff: Distance from border to consider "near border"
        require_start_near_border: Require start point near border
        require_end_near_border: Require end point near border
        min_endpoint_near_border: Minimum number of endpoints that must be near border
            (0=no requirement, 1=at least one, 2=both)

    Returns:
        Filtered binary mask

    """
    if border_standoff is None:
        border_standoff = GT_PARAMS["border_standoff"]

    labeled, num_features = scipy_label(mask)
    result = np.zeros_like(mask)

    for i in range(1, num_features + 1):
        component = labeled == i
        start_slice, end_slice, z_indices = _get_component_endpoints(component)

        if start_slice is None:
            continue

        # Check if endpoints are near borders
        start_touches = _check_touches_border(start_slice, border_standoff)
        end_touches = _check_touches_border(end_slice, border_standoff)

        # Count how many endpoints are near border
        endpoints_near_border = int(start_touches["any"]) + int(end_touches["any"])

        # Apply filtering criteria
        keep = True
        if require_start_near_border and not start_touches["any"]:
            keep = False
        if require_end_near_border and not end_touches["any"]:
            keep = False
        if endpoints_near_border < min_endpoint_near_border:
            keep = False

        if keep:
            result[component] = 1

    return result


def filter_by_border_connectivity_soft(
    mask: np.ndarray,
    pr: np.ndarray,
    border_standoff: int = None,
    border_boost: float = 0.3,
) -> np.ndarray:
    """Soft filtering: boost probability of components with border connectivity.

    Instead of hard filtering, this method boosts the probability scores
    of voxels in components that have proper border connectivity.

    Args:
        mask: Initial binary mask
        pr: Original probability map
        border_standoff: Distance from border to consider "near border"
        border_boost: Probability boost for border-connected components

    Returns:
        Binary mask with border-connected components more likely to survive

    """
    if border_standoff is None:
        border_standoff = GT_PARAMS["border_standoff"]

    labeled, num_features = scipy_label(mask)
    boosted_pr = pr.copy()

    for i in range(1, num_features + 1):
        component = labeled == i
        start_slice, end_slice, z_indices = _get_component_endpoints(component)

        if start_slice is None:
            continue

        start_touches = _check_touches_border(start_slice, border_standoff)
        end_touches = _check_touches_border(end_slice, border_standoff)

        # Boost probability for components with good border connectivity
        if start_touches["any"] and end_touches["any"]:
            boosted_pr[component] += border_boost
        elif start_touches["any"] or end_touches["any"]:
            boosted_pr[component] += border_boost / 2

    # Re-threshold with boosted probabilities
    return (boosted_pr > 0.5).astype(np.uint8) & mask


def extend_ribbons_to_borders(
    mask: np.ndarray,
    pr: np.ndarray,
    border_standoff: int = None,
    extension_threshold: float = 0.1,
    max_extension_distance: int = 30,
) -> np.ndarray:
    """Extend ribbon endpoints toward nearest borders using probability guidance.

    Since ribbons should start/end near borders, we can try to extend
    partial ribbons that are close to but not quite reaching the border.

    Args:
        mask: Binary mask
        pr: Probability map (used to guide extension)
        border_standoff: Distance from border to consider "at border"
        extension_threshold: Minimum probability to extend into
        max_extension_distance: Maximum distance to extend

    Returns:
        Extended binary mask

    """
    if border_standoff is None:
        border_standoff = GT_PARAMS["border_standoff"]

    z_depth, h, w = mask.shape
    result = mask.copy()
    labeled, num_features = scipy_label(mask)

    for i in range(1, num_features + 1):
        component = labeled == i
        start_slice, end_slice, z_indices = _get_component_endpoints(component)

        if start_slice is None or len(z_indices) < 10:
            continue

        # Check if start/end need extension
        start_touches = _check_touches_border(start_slice, border_standoff)
        end_touches = _check_touches_border(end_slice, border_standoff)

        # Try to extend start if not touching border
        if not start_touches["any"]:
            z_start = z_indices[0]
            # Extend backward in z
            for z_ext in range(z_start - 1, max(0, z_start - max_extension_distance), -1):
                # Find centroid of current start
                coords = np.array(
                    np.where(result[z_ext + 1] & component[z_ext + 1 : z_ext + 2].any(axis=0)),
                ).T
                if len(coords) == 0:
                    coords = np.array(np.where(start_slice)).T
                if len(coords) == 0:
                    break
                centroid = coords.mean(axis=0).astype(int)

                # Dilate around centroid using probability guidance
                y, x = centroid
                y_min, y_max = max(0, y - 5), min(h, y + 6)
                x_min, x_max = max(0, x - 5), min(w, x + 6)

                local_pr = pr[z_ext, y_min:y_max, x_min:x_max]
                extension = local_pr > extension_threshold

                if extension.any():
                    result[z_ext, y_min:y_max, x_min:x_max] |= extension.astype(np.uint8)
                else:
                    break

        # Try to extend end if not touching border
        if not end_touches["any"]:
            z_end = z_indices[-1]
            # Extend forward in z
            for z_ext in range(z_end + 1, min(z_depth, z_end + max_extension_distance)):
                coords = np.array(
                    np.where(result[z_ext - 1] & component[z_ext - 1 : z_ext].any(axis=0)),
                ).T
                if len(coords) == 0:
                    coords = np.array(np.where(end_slice)).T
                if len(coords) == 0:
                    break
                centroid = coords.mean(axis=0).astype(int)

                y, x = centroid
                y_min, y_max = max(0, y - 5), min(h, y + 6)
                x_min, x_max = max(0, x - 5), min(w, x + 6)

                local_pr = pr[z_ext, y_min:y_max, x_min:x_max]
                extension = local_pr > extension_threshold

                if extension.any():
                    result[z_ext, y_min:y_max, x_min:x_max] |= extension.astype(np.uint8)
                else:
                    break

    return result


# =============================================================================
# PATH CONTINUITY AND TRAJECTORY ENFORCEMENT
# =============================================================================


def compute_centroid_path(component: np.ndarray) -> np.ndarray:
    """Compute the centroid path of a component through z-slices.

    Args:
        component: 3D binary mask of a single component

    Returns:
        Array of shape (N, 3) with (z, y, x) centroids for each z-slice

    """
    z_presence = component.any(axis=(1, 2))
    z_indices = np.where(z_presence)[0]

    if len(z_indices) == 0:
        return np.array([])

    centroids = []
    for z in z_indices:
        coords = np.array(np.where(component[z])).T
        if len(coords) > 0:
            centroid = coords.mean(axis=0)
            centroids.append([z, centroid[0], centroid[1]])

    return np.array(centroids)


def filter_by_path_smoothness(
    mask: np.ndarray,
    max_displacement_per_slice: float = None,
    min_path_smoothness: float = 0.3,
) -> np.ndarray:
    """Filter components by path smoothness (reject erratic trajectories).

    GT analysis shows ribbons follow smooth paths with:
    - Mean displacement: 2.0 voxels/slice
    - Max displacement: 6.0 voxels/slice
    - Path smoothness: 0.487

    Args:
        mask: Binary mask
        max_displacement_per_slice: Maximum allowed displacement between slices
        min_path_smoothness: Minimum required path smoothness score

    Returns:
        Filtered binary mask

    """
    if max_displacement_per_slice is None:
        max_displacement_per_slice = GT_PARAMS["max_centroid_displacement"]

    labeled, num_features = scipy_label(mask)
    result = np.zeros_like(mask)

    for i in range(1, num_features + 1):
        component = labeled == i
        centroids = compute_centroid_path(component)

        if len(centroids) < 3:
            # Too short to evaluate, keep it
            result[component] = 1
            continue

        # Compute displacements between consecutive centroids
        xy_coords = centroids[:, 1:]  # (y, x) coordinates
        displacements = np.sqrt(np.sum(np.diff(xy_coords, axis=0) ** 2, axis=1))

        # Check max displacement
        max_disp = displacements.max() if len(displacements) > 0 else 0

        # Compute smoothness (inverse of displacement variance)
        disp_std = displacements.std() if len(displacements) > 1 else 0
        smoothness = 1.0 / (1.0 + disp_std)

        # Keep if path is smooth enough
        if max_disp <= max_displacement_per_slice * 2 and smoothness >= min_path_smoothness:
            result[component] = 1

    return result


def remove_spurious_branches(
    mask: np.ndarray,
    min_branch_size: int = 500,
    branch_angle_threshold: float = 45.0,
) -> np.ndarray:
    """Remove spurious branches from ribbon-like structures.

    GT analysis: Ribbons should NOT bifurcate. Any branches are likely noise.

    This function detects and removes small protrusions that branch off
    from the main ribbon path.

    Args:
        mask: Binary mask
        min_branch_size: Minimum size of branch to consider keeping
        branch_angle_threshold: Angle threshold for detecting branches (degrees)

    Returns:
        Cleaned binary mask

    """
    # Use erosion to find the core, then use it to identify branches
    struct = ball(1)
    core = binary_erosion(mask, structure=struct, iterations=2)

    # Label the eroded core
    labeled_core, num_cores = scipy_label(core)

    # For each original component, keep only the largest connected core
    labeled_orig, num_features = scipy_label(mask)
    result = np.zeros_like(mask)

    for i in range(1, num_features + 1):
        component = labeled_orig == i
        component_core = core & component

        if component_core.sum() == 0:
            # No core found, keep original if large enough
            if component.sum() >= min_branch_size:
                result[component] = 1
            continue

        # Find the largest core within this component
        core_labels = np.unique(labeled_core[component_core])
        core_labels = core_labels[core_labels != 0]

        if len(core_labels) == 0:
            if component.sum() >= min_branch_size:
                result[component] = 1
            continue

        # Keep the largest core and dilate back to original boundaries
        largest_core_label = max(core_labels, key=lambda l: (labeled_core == l).sum())
        largest_core = labeled_core == largest_core_label

        # Dilate the core back to approximate original size, but constrained to original mask
        dilated = binary_dilation(largest_core, structure=struct, iterations=3)
        result[component & dilated] = 1

    return result


# =============================================================================
# GT-INFORMED HOLE FILLING
# =============================================================================


def fill_holes_gt(
    mask: np.ndarray,
    max_hole_size: int = None,
    fill_2d: bool = True,
    fill_3d: bool = True,
) -> np.ndarray:
    """Fill holes based on GT hole characteristics.

    GT analysis: Only 2.3% have holes, max_hole_size=100 is safe.

    Args:
        mask: Binary mask
        max_hole_size: Maximum hole size to fill (default from GT)
        fill_2d: Fill holes per z-slice
        fill_3d: Fill 3D holes

    Returns:
        Hole-filled mask

    """
    if max_hole_size is None:
        max_hole_size = GT_PARAMS["max_hole_size"]

    result = mask.copy()

    # 2D hole filling per slice (catches most holes)
    if fill_2d:
        for z in range(result.shape[0]):
            result[z] = binary_fill_holes(result[z]).astype(result.dtype)

    # 3D hole filling for small holes
    if fill_3d:
        result = remove_small_holes(
            result.astype(bool),
            area_threshold=max_hole_size,
        ).astype(np.uint8)

    return result


def fill_holes_component_aware(
    mask: np.ndarray,
    max_hole_size_2d: int = 50,
    max_hole_size_3d: int = 100,
    min_ribbon_separation: float = 1.5,
    connectivity: int = 26,
) -> np.ndarray:
    """Component-aware hole filling that respects ribbon boundaries.

    PROBLEM: Standard hole filling can bridge nearby ribbons (1-2 voxels apart).
    SOLUTION: Fill holes ONLY within each connected component, never across components.

    This is critical for thin (3-4 voxel) ribbons that may be very close together.
    The algorithm:
    1. Label connected components
    2. For each component, create a bounding box mask
    3. Fill 2D holes per z-slice WITHIN the component's region only
    4. Fill small 3D holes WITHIN the component only
    5. Verify no bridging occurred via distance transform check

    Args:
        mask: Binary mask (Z, H, W)
        max_hole_size_2d: Maximum 2D hole area to fill per slice
        max_hole_size_3d: Maximum 3D hole volume to fill
        min_ribbon_separation: Minimum distance between ribbons (from GT: 1.4)
        connectivity: cc3d connectivity (6, 18, or 26)

    Returns:
        Hole-filled mask that doesn't bridge separate components

    """
    if mask.sum() == 0:
        return mask

    # Get initial labels
    labels_before = cc3d.connected_components(mask, connectivity=connectivity)
    num_labels_before = labels_before.max()

    result = np.zeros_like(mask)

    # Process each component independently
    for label_id in range(1, num_labels_before + 1):
        component = labels_before == label_id

        if component.sum() == 0:
            continue

        # Get bounding box for efficiency
        z_indices = np.where(component.any(axis=(1, 2)))[0]
        if len(z_indices) == 0:
            continue

        z_min, z_max = z_indices[0], z_indices[-1] + 1
        y_indices = np.where(component.any(axis=(0, 2)))[0]
        x_indices = np.where(component.any(axis=(0, 1)))[0]
        if len(y_indices) == 0 or len(x_indices) == 0:
            continue

        y_min, y_max = y_indices[0], y_indices[-1] + 1
        x_min, x_max = x_indices[0], x_indices[-1] + 1

        # Add small padding for boundary effects
        pad = 2
        z_min_p = max(0, z_min - pad)
        z_max_p = min(mask.shape[0], z_max + pad)
        y_min_p = max(0, y_min - pad)
        y_max_p = min(mask.shape[1], y_max + pad)
        x_min_p = max(0, x_min - pad)
        x_max_p = min(mask.shape[2], x_max + pad)

        # Extract component in bounding box
        comp_crop = component[z_min_p:z_max_p, y_min_p:y_max_p, x_min_p:x_max_p].copy()

        # Fill 2D holes per slice (constrained to component region)
        for z in range(comp_crop.shape[0]):
            slice_2d = comp_crop[z]
            if slice_2d.sum() == 0:
                continue

            # Fill holes in this slice
            filled = binary_fill_holes(slice_2d)

            # Only keep filled regions that are small enough
            # (large holes might be real gaps between ribbons)
            holes = filled & ~slice_2d
            if holes.sum() > 0 and holes.sum() <= max_hole_size_2d:
                comp_crop[z] = filled.astype(comp_crop.dtype)

        # Fill small 3D holes within component
        try:
            # Use remove_small_holes on the component crop
            comp_filled = remove_small_holes(
                comp_crop.astype(bool),
                area_threshold=max_hole_size_3d,
            ).astype(np.uint8)
            comp_crop = comp_filled
        except Exception:
            pass  # If 3D fill fails, keep 2D-filled result

        # Put back into result
        result[z_min_p:z_max_p, y_min_p:y_max_p, x_min_p:x_max_p] |= comp_crop

    # SAFETY CHECK: Verify we didn't accidentally bridge components
    labels_after = cc3d.connected_components(result, connectivity=connectivity)
    num_labels_after = labels_after.max()

    if num_labels_after < num_labels_before:
        # We accidentally merged some components - revert to original
        # This shouldn't happen with proper component-isolated processing,
        # but is a safety net
        return mask

    return result


def fill_holes_slice_constrained(
    mask: np.ndarray,
    max_hole_size: int = 30,
    max_hole_ratio: float = 0.5,
) -> np.ndarray:
    """Conservative 2D slice-by-slice hole filling with size constraints.

    Fills holes only if they are:
    1. Small enough (below max_hole_size)
    2. Surrounded by component (high hole_ratio indicates internal hole)

    This is safer than binary_fill_holes for thin structures near each other.

    Args:
        mask: Binary mask
        max_hole_size: Maximum hole area in pixels
        max_hole_ratio: Maximum ratio of hole to component area per slice

    Returns:
        Conservatively hole-filled mask

    """
    result = mask.copy()

    for z in range(result.shape[0]):
        slice_2d = result[z]
        if slice_2d.sum() == 0:
            continue

        # Fill all holes first
        filled = binary_fill_holes(slice_2d)
        holes = filled.astype(np.uint8) - slice_2d

        if holes.sum() == 0:
            continue

        # Label individual holes
        hole_labels, num_holes = scipy_label(holes)

        # Only fill small holes
        for hole_id in range(1, num_holes + 1):
            hole_mask = hole_labels == hole_id
            hole_size = hole_mask.sum()

            if hole_size <= max_hole_size:
                # Additional check: hole should be mostly surrounded by foreground
                # Dilate hole and check how much overlaps with original mask
                from scipy.ndimage import binary_dilation as bd

                dilated_hole = bd(hole_mask, iterations=1)
                boundary = dilated_hole & ~hole_mask
                if boundary.sum() > 0:
                    foreground_ratio = (boundary & slice_2d.astype(bool)).sum() / boundary.sum()
                    if foreground_ratio >= 0.6:  # At least 60% surrounded
                        result[z][hole_mask] = 1

    return result


def fill_holes_distance_constrained(
    mask: np.ndarray,
    max_fill_distance: float = 2.0,
    max_hole_size: int = 50,
) -> np.ndarray:
    """Fill holes only near existing foreground (distance constrained).

    Uses distance transform to ensure filled regions are close to
    existing foreground, preventing bridging across large gaps.

    Args:
        mask: Binary mask
        max_fill_distance: Maximum distance from foreground to fill
        max_hole_size: Maximum hole size to consider

    Returns:
        Distance-constrained hole-filled mask

    """
    result = mask.copy()

    # Compute distance from foreground
    dist_from_fg = distance_transform_edt(~mask.astype(bool))

    # Fill 2D holes per slice
    for z in range(result.shape[0]):
        slice_2d = result[z]
        if slice_2d.sum() == 0:
            continue

        filled = binary_fill_holes(slice_2d)
        holes = filled & ~slice_2d.astype(bool)

        if holes.sum() == 0:
            continue

        # Only fill holes that are close to foreground
        close_to_fg = dist_from_fg[z] <= max_fill_distance
        safe_fill = holes & close_to_fg

        # Also check hole is small
        hole_labels, num_holes = scipy_label(holes)
        for hole_id in range(1, num_holes + 1):
            hole_mask = hole_labels == hole_id
            if hole_mask.sum() <= max_hole_size:
                if (hole_mask & close_to_fg).sum() / hole_mask.sum() >= 0.8:
                    result[z][hole_mask] = 1

    return result


def fill_gaps_probability_guided(
    mask: np.ndarray,
    pr: np.ndarray,
    low_threshold: float = 0.05,
    dilation_radius: int = 2,
    max_gap_size: int = 100,
) -> np.ndarray:
    """Fill gaps in the mask using probability guidance.

    PROBLEM: Standard hole filling only works on fully enclosed holes.
    Thin ribbons often have "weak" regions that don't form closed holes
    but should still be part of the ribbon.

    SOLUTION: Use the probability map to recover weak voxels that are:
    1. Close to existing foreground (within dilation radius)
    2. Have non-trivial probability (above low_threshold)
    3. Form small connected regions (not large false positives)

    This is like hysteresis thresholding but applied AFTER initial segmentation
    to recover weak regions within ribbons.

    Args:
        mask: Initial binary mask
        pr: Probability map (0-1)
        low_threshold: Minimum probability to consider for gap filling
        dilation_radius: How far from existing foreground to look
        max_gap_size: Maximum size of gaps to fill (per component)

    Returns:
        Gap-filled mask

    """
    if mask.sum() == 0:
        return mask

    result = mask.copy()

    # Create a "potential fill" zone: dilated mask minus original mask
    struct = ball(dilation_radius)
    dilated = binary_dilation(mask, structure=struct)
    potential_zone = dilated & ~mask.astype(bool)

    # Find voxels in the potential zone with sufficient probability
    candidates = potential_zone & (pr >= low_threshold)

    if candidates.sum() == 0:
        return result

    # Label candidate regions and filter by size
    candidate_labels, num_candidates = scipy_label(candidates)

    for label_id in range(1, num_candidates + 1):
        region = candidate_labels == label_id
        region_size = region.sum()

        if region_size <= max_gap_size:
            # Additional check: region should be touching the original mask
            dilated_region = binary_dilation(region, iterations=1)
            if (dilated_region & mask.astype(bool)).any():
                result[region] = 1

    return result


def fill_weak_ribbon_regions(
    mask: np.ndarray,
    pr: np.ndarray,
    low_threshold: float = 0.08,
    ribbon_dilation: int = 3,
    min_prob_boost: float = 0.1,
) -> np.ndarray:
    """Aggressively fill weak regions within ribbon structures.

    Uses morphological closing on a probability-boosted version of the mask
    to recover thin gaps and weak regions.

    The key insight: thin ribbons may have regions where probability drops
    slightly below threshold, creating artificial breaks. We can recover
    these by looking at probability in a neighborhood.

    Args:
        mask: Initial binary mask
        pr: Probability map
        low_threshold: Minimum probability for region recovery
        ribbon_dilation: Dilation radius for morphological operations
        min_prob_boost: Minimum probability boost from neighbors

    Returns:
        Enhanced mask with filled weak regions

    """
    if mask.sum() == 0:
        return mask

    # Step 1: Create a "soft" mask based on probability
    # Voxels get added if they have decent probability AND are near existing foreground
    struct = ball(ribbon_dilation)
    dilated = binary_dilation(mask, structure=struct)

    # Soft candidates: in dilated zone, have some probability
    soft_candidates = dilated & (pr >= low_threshold) & ~mask.astype(bool)

    # Step 2: For each candidate voxel, check if neighbors in original mask
    # have high probability (indicates we're inside a ribbon, not bridging)
    result = mask.copy()

    if soft_candidates.sum() > 0:
        # Use a smoothed probability to determine "inside ribbon" regions
        from scipy.ndimage import uniform_filter

        smoothed_pr = uniform_filter(pr, size=3)

        # Candidates that are likely inside ribbons (smoothed prob is high)
        likely_inside = soft_candidates & (smoothed_pr >= low_threshold + min_prob_boost)

        result = result | likely_inside.astype(np.uint8)

    # Step 3: Apply morphological closing to fill remaining small gaps
    # Use anisotropic structuring element (ribbons extend along z)
    close_struct = np.zeros((3, 5, 5), dtype=bool)
    close_struct[1] = disk(2)  # Main closing in xy
    close_struct[0] = close_struct[2] = disk(1)  # Smaller in z

    result = binary_closing(result, structure=close_struct)

    return result.astype(np.uint8)


# =============================================================================
# GT-INFORMED MORPHOLOGICAL OPERATIONS
# =============================================================================


def morphology_gt(
    mask: np.ndarray,
    closing_size: int = None,
    opening_size: int = None,
    anisotropic: bool = True,
) -> np.ndarray:
    """Morphological cleanup based on GT ribbon width.

    GT ribbon width: 4.0 ± 0.3 voxels
    Recommended closing/opening size: 1

    Args:
        mask: Binary mask
        closing_size: Size for closing operation
        opening_size: Size for opening operation
        anisotropic: Use anisotropic structuring element (larger in xy)

    Returns:
        Cleaned mask

    """
    if closing_size is None:
        closing_size = GT_PARAMS["closing_size"]
    if opening_size is None:
        opening_size = GT_PARAMS["opening_size"]

    result = mask.astype(bool)

    if closing_size > 0:
        if anisotropic:
            # Larger in xy plane (ribbons extend along z)
            struct = np.zeros(
                (closing_size, closing_size * 2 + 1, closing_size * 2 + 1),
                dtype=bool,
            )
            for z in range(closing_size):
                struct[z] = disk(closing_size)
        else:
            struct = ball(closing_size)
        result = binary_closing(result, structure=struct)

    if opening_size > 0:
        if anisotropic:
            struct = np.zeros(
                (opening_size, opening_size * 2 + 1, opening_size * 2 + 1),
                dtype=bool,
            )
            for z in range(opening_size):
                struct[z] = disk(opening_size)
        else:
            struct = ball(opening_size)
        result = binary_opening(result, structure=struct)

    return result.astype(np.uint8)


def smooth_boundaries_gt(
    mask: np.ndarray,
    sigma: float = 0.5,
    threshold: float = 0.5,
) -> np.ndarray:
    """Smooth mask boundaries using Gaussian filtering.

    Based on GT surface roughness: 0.451
    This helps improve surface dice score.

    Args:
        mask: Binary mask
        sigma: Gaussian smoothing sigma
        threshold: Threshold for re-binarization

    Returns:
        Smoothed binary mask

    """
    smoothed = gaussian_filter(mask.astype(np.float32), sigma=sigma)
    return (smoothed > threshold).astype(np.uint8)


# =============================================================================
# GT-INFORMED Z-CONTINUITY
# =============================================================================


def enforce_z_continuity_gt(
    mask: np.ndarray,
    min_z_span: int = None,
    min_overlap_ratio: float = None,
    max_gap: int = 2,
) -> np.ndarray:
    """Enforce z-continuity based on GT characteristics.

    GT: mean IoU overlap 0.828, min overlap ratio 0.23

    Args:
        mask: Binary mask
        min_z_span: Minimum z-span required
        min_overlap_ratio: Minimum IoU between consecutive slices
        max_gap: Maximum allowed gap in z-slices

    Returns:
        Z-continuity enforced mask

    """
    if min_z_span is None:
        min_z_span = GT_PARAMS["min_z_span"]
    if min_overlap_ratio is None:
        min_overlap_ratio = GT_PARAMS["min_overlap_ratio"]

    labeled, num_features = scipy_label(mask)
    result = np.zeros_like(mask)

    for i in range(1, num_features + 1):
        component = labeled == i
        z_indices = np.where(component.any(axis=(1, 2)))[0]

        if len(z_indices) < min_z_span:
            continue

        # Check for gaps
        has_large_gap = False
        for j in range(len(z_indices) - 1):
            if z_indices[j + 1] - z_indices[j] > max_gap:
                has_large_gap = True
                break

        if has_large_gap:
            continue

        # Check slice overlap (sample for speed)
        good_overlap = True
        sample_indices = z_indices[:: max(1, len(z_indices) // 20)]  # Sample ~20 pairs

        for j in range(len(sample_indices) - 1):
            z1, z2 = sample_indices[j], sample_indices[j + 1]
            if z2 - z1 > max_gap:
                continue

            slice1 = component[z1]
            slice2 = component[z2]
            intersection = np.logical_and(slice1, slice2).sum()
            union = np.logical_or(slice1, slice2).sum()

            if union > 0 and intersection / union < min_overlap_ratio:
                good_overlap = False
                break

        if good_overlap:
            result[component] = 1

    return result


# =============================================================================
# GT-INFORMED INSTANCE SEPARATION
# =============================================================================


def enforce_ribbon_separation(
    mask: np.ndarray,
    pr: np.ndarray,
    min_separation: float = None,
) -> np.ndarray:
    """Ensure ribbons don't touch (GT: 0% touching).

    Uses watershed on distance transform to separate potentially
    merged ribbons.

    Args:
        mask: Binary mask
        pr: Probability map (used for watershed landscape)
        min_separation: Minimum expected separation (from GT)

    Returns:
        Separated binary mask

    """
    if min_separation is None:
        min_separation = GT_PARAMS["min_separation"]

    if mask.sum() == 0:
        return mask

    # Distance transform
    distance = distance_transform_edt(mask)

    # Find local maxima as markers (ribbon centers)
    local_max = (distance == maximum_filter(distance, size=5)) & (distance > min_separation)
    markers, num_markers = scipy_label(local_max)

    if num_markers <= 1:
        return mask

    # Watershed using negative probability as landscape
    landscape = -pr * mask
    labels = watershed(landscape, markers, mask=mask)

    # Convert back to binary
    return (labels > 0).astype(np.uint8)


def separate_by_erosion_dilation(
    mask: np.ndarray,
    erosion_radius: int = None,
) -> np.ndarray:
    """Separate merged ribbons using erosion-dilation.

    GT: safe erosion radius = 1 (min separation 1.4 voxels)

    Args:
        mask: Binary mask
        erosion_radius: Erosion radius (from GT)

    Returns:
        Separated mask with original boundaries restored

    """
    if erosion_radius is None:
        erosion_radius = GT_PARAMS["safe_erosion_radius"]

    # Erode to separate
    struct = ball(erosion_radius)
    eroded = binary_erosion(mask, structure=struct)

    # Label separated components
    labeled, num_features = scipy_label(eroded)

    # Dilate each component separately to restore boundaries
    result = np.zeros_like(mask)
    for i in range(1, num_features + 1):
        component = (labeled == i).astype(bool)
        # Dilate back but constrain to original mask
        dilated = binary_dilation(component, structure=struct)
        dilated = dilated & mask.astype(bool)
        result = np.maximum(result, dilated.astype(np.uint8))

    return result


# =============================================================================
# GT-INFORMED TOPOLOGY CLEANUP
# =============================================================================


def remove_topology_defects(
    mask: np.ndarray,
    remove_tunnels: bool = True,
    remove_voids: bool = True,
    max_defect_size: int = 50,
) -> np.ndarray:
    """Remove topological defects (tunnels/voids).

    GT: 0.4% have tunnels, 0.0% have voids

    Args:
        mask: Binary mask
        remove_tunnels: Fill tunnel-like holes
        remove_voids: Fill void-like cavities
        max_defect_size: Maximum size of defects to remove

    Returns:
        Topologically cleaned mask

    """
    result = mask.copy()

    if remove_voids:
        # Fill small 3D holes (voids)
        result = remove_small_holes(
            result.astype(bool),
            area_threshold=max_defect_size,
        ).astype(np.uint8)

    if remove_tunnels:
        # For tunnels, fill 2D holes slice by slice
        for z in range(result.shape[0]):
            result[z] = binary_fill_holes(result[z]).astype(result.dtype)

    return result


# =============================================================================
# FRAGMENTATION REPAIR
# =============================================================================


def reconnect_fragments(
    mask: np.ndarray,
    max_gap: int = None,
    min_component_size: int = 100,
) -> np.ndarray:
    """Reconnect fragmented ribbon segments.

    GT fragmentation rate: 4.51%

    Args:
        mask: Binary mask
        max_gap: Maximum gap to bridge (from GT)
        min_component_size: Minimum size of components to connect

    Returns:
        Reconnected mask

    """
    if max_gap is None:
        max_gap = GT_PARAMS["fragmentation_gap"]

    # Dilate to bridge small gaps
    struct = ball(max_gap)
    dilated = binary_dilation(mask, structure=struct)

    # Find connected regions in dilated mask
    labeled_dilated, _ = scipy_label(dilated)

    # For each connected region in dilated, keep original pixels
    result = np.zeros_like(mask)

    labeled_orig, num_features = scipy_label(mask)

    for i in range(1, num_features + 1):
        component = labeled_orig == i
        if component.sum() >= min_component_size:
            result[component] = 1

    # Also include any original mask pixels that got connected
    result = result | (mask & (labeled_dilated > 0))

    return result.astype(np.uint8)


# =============================================================================
# COMPOSITE GT-INFORMED PIPELINES
# =============================================================================


def postprocess_gt_minimal(
    pr: np.ndarray,
    threshold: float = 0.20,
) -> np.ndarray:
    """Minimal GT-informed post-processing.

    Light touch: threshold + 2D hole filling only.
    Best for preserving surface dice.
    """
    mask = threshold_gt_optimized(pr, threshold)
    mask = fill_holes_gt(mask, fill_2d=True, fill_3d=False)
    return mask


def postprocess_gt_balanced(
    pr: np.ndarray,
    threshold: float = 0.18,
    min_size: int = 500,
) -> np.ndarray:
    """Balanced GT-informed post-processing.

    Good balance between cleanup and preservation.
    """
    mask = threshold_gt_optimized(pr, threshold)
    mask = fill_holes_gt(mask, fill_2d=True, fill_3d=True)
    mask = remove_small_gt(mask, min_size=min_size)
    return mask


def postprocess_gt_topology(
    pr: np.ndarray,
    low_threshold: float = 0.12,
    high_threshold: float = 0.28,
    min_size: int = 1000,
    min_z_span: int = 30,
) -> np.ndarray:
    """Topology-focused GT-informed post-processing.

    Optimizes for Betti matching and VOI.
    """
    mask = hysteresis_gt_informed(pr, low_threshold, high_threshold)
    mask = fill_holes_gt(mask, fill_2d=True, fill_3d=True)
    mask = remove_topology_defects(mask)
    mask = remove_small_gt(mask, min_size=min_size)
    mask = enforce_z_continuity_gt(mask, min_z_span=min_z_span)
    return mask


def postprocess_gt_strict(
    pr: np.ndarray,
    low_threshold: float = 0.15,
    high_threshold: float = 0.30,
    min_size: int = 3000,
    min_z_span: int = 50,
) -> np.ndarray:
    """Strict GT-informed post-processing.

    Aggressive filtering, keeps only high-confidence ribbons.
    """
    mask = hysteresis_gt_informed(pr, low_threshold, high_threshold)
    mask = fill_holes_gt(mask)
    mask = morphology_gt(mask)
    mask = remove_topology_defects(mask)
    mask = remove_small_gt(mask, min_size=min_size)
    mask = remove_fragments_by_z_span(mask, min_z_span=min_z_span)
    mask = enforce_z_continuity_gt(mask)
    return mask


def postprocess_gt_surface_optimized(
    pr: np.ndarray,
    threshold: float = 0.19,
    smooth_sigma: float = 0.3,
) -> np.ndarray:
    """Surface dice optimized GT-informed post-processing.

    Focus on smooth boundaries to maximize surface dice.
    """
    mask = threshold_gt_optimized(pr, threshold)
    mask = fill_holes_gt(mask, fill_2d=True, fill_3d=False)
    mask = smooth_boundaries_gt(mask, sigma=smooth_sigma)
    return mask


def postprocess_gt_voi_optimized(
    pr: np.ndarray,
    threshold: float = 0.22,
    min_size: int = 500,
) -> np.ndarray:
    """VOI optimized GT-informed post-processing.

    Focus on correct instance separation (reduce merge/split errors).
    Higher threshold reduces false positives (merge errors).
    """
    mask = threshold_gt_optimized(pr, threshold)
    mask = fill_holes_gt(mask, fill_2d=True, fill_3d=False)
    mask = remove_small_gt(mask, min_size=min_size)
    # Don't do separation - it can hurt VOI if it over-segments
    return mask


def postprocess_gt_comprehensive(
    pr: np.ndarray,
    low_threshold: float = 0.12,
    high_threshold: float = 0.25,
    min_size: int = 1500,
    min_z_span: int = 40,
    use_morphology: bool = True,
) -> np.ndarray:
    """Comprehensive GT-informed post-processing.

    Full pipeline using all GT priors.
    """
    # Hysteresis for robust thresholding
    mask = hysteresis_gt_informed(pr, low_threshold, high_threshold)

    # Fill holes (GT: 2.3% have holes)
    mask = fill_holes_gt(mask, fill_2d=True, fill_3d=True)

    # Morphological cleanup (GT: width 4.0, closing/opening 1)
    if use_morphology:
        mask = morphology_gt(mask)

    # Remove topology defects (GT: 0.4% tunnels, 0% voids)
    mask = remove_topology_defects(mask)

    # Size filtering (GT: 5th pct = 6100)
    mask = remove_small_gt(mask, min_size=min_size)

    # Z-continuity (GT: mean overlap 0.828)
    mask = enforce_z_continuity_gt(mask, min_z_span=min_z_span)

    return mask


# =============================================================================
# NEW: BORDER-AWARE PIPELINES (leveraging 99.6% border connectivity prior)
# =============================================================================


def postprocess_border_aware(
    pr: np.ndarray,
    threshold: float = 0.18,
    min_size: int = 500,
    border_standoff: int = None,
    require_both_endpoints: bool = True,
) -> np.ndarray:
    """Border-aware post-processing using the key GT insight.

    KEY INSIGHT: 99.6% of GT ribbons start AND end near XY borders.
    This filter removes false positives that don't follow this pattern.

    Args:
        pr: Probability map
        threshold: Initial threshold
        min_size: Minimum component size
        border_standoff: Distance from border to consider "near border"
        require_both_endpoints: If True, require BOTH endpoints near border

    """
    if border_standoff is None:
        border_standoff = GT_PARAMS["border_standoff"]

    # Initial thresholding and basic cleanup
    mask = threshold_gt_optimized(pr, threshold)
    mask = fill_holes_gt(mask, fill_2d=True, fill_3d=True)
    mask = remove_small_gt(mask, min_size=min_size)

    # KEY: Filter by border endpoint connectivity
    min_endpoints = 2 if require_both_endpoints else 1
    mask = filter_by_border_endpoints(
        mask,
        border_standoff=border_standoff,
        require_start_near_border=require_both_endpoints,
        require_end_near_border=require_both_endpoints,
        min_endpoint_near_border=min_endpoints,
    )

    return mask


def postprocess_hole_fill_safe(
    pr: np.ndarray,
    threshold: float = 0.15,
    min_size: int = 25,
    border_standoff: int = None,
    require_both_endpoints: bool = False,
    max_hole_size_2d: int = 50,
    max_hole_size_3d: int = 100,
    use_distance_constraint: bool = True,
    max_fill_distance: float = 2.0,
    low_prob_threshold: float = 0.05,
    use_probability_guided: bool = True,
) -> np.ndarray:
    """Safe hole-filling for thin ribbons using probability-guided gap recovery.

    PROBLEM: Thin ribbons (3-4 voxels) can have weak regions where probability
    drops below threshold, creating artificial breaks. Standard binary hole
    filling doesn't help because these aren't enclosed holes.

    SOLUTION: Probability-guided gap filling that:
    1. Starts with initial thresholding
    2. Recovers weak voxels near existing foreground using probability map
    3. Uses morphological closing to fill remaining small gaps
    4. Applies border filtering as final cleanup

    Based on GT analysis:
    - Ribbon width: 4.0 ± 0.3 voxels
    - Min separation: 1.4 voxels
    - Only 2.3% have holes, but weak regions are more common

    Args:
        pr: Probability map
        threshold: Initial threshold
        min_size: Minimum component size to keep
        border_standoff: Distance from border for endpoint filtering
        require_both_endpoints: If True, require BOTH endpoints near border
        max_hole_size_2d: Maximum 2D hole area to fill per slice
        max_hole_size_3d: Maximum 3D hole volume to fill
        use_distance_constraint: If True, use distance-constrained filling
        max_fill_distance: Maximum distance from foreground to fill
        low_prob_threshold: Lower probability threshold for gap recovery
        use_probability_guided: Use probability-guided gap filling (recommended)

    Returns:
        Binary mask with recovered weak regions

    """
    if border_standoff is None:
        border_standoff = GT_PARAMS["border_standoff"]

    # Initial thresholding
    mask = threshold_gt_optimized(pr, threshold)

    # Apply probability-guided gap filling
    if use_probability_guided:
        # First recover weak regions using probability
        mask = fill_gaps_probability_guided(
            mask,
            pr,
            low_threshold=low_prob_threshold,
            dilation_radius=2,
            max_gap_size=max_hole_size_3d,
        )

        # Then fill remaining weak regions more aggressively
        mask = fill_weak_ribbon_regions(
            mask,
            pr,
            low_threshold=low_prob_threshold,
            ribbon_dilation=3,
            min_prob_boost=0.05,
        )

    # Also do standard 2D hole filling for any enclosed holes
    mask = fill_holes_gt(mask, fill_2d=True, fill_3d=True, max_hole_size=max_hole_size_3d)

    # Size filtering
    mask = remove_small_gt(mask, min_size=min_size)

    # Border endpoint filtering
    min_endpoints = 2 if require_both_endpoints else 1
    mask = filter_by_border_endpoints(
        mask,
        border_standoff=border_standoff,
        require_start_near_border=require_both_endpoints,
        require_end_near_border=require_both_endpoints,
        min_endpoint_near_border=min_endpoints,
    )

    return mask


def postprocess_hole_fill_conservative(
    pr: np.ndarray,
    threshold: float = 0.15,
    min_size: int = 25,
    border_standoff: int = None,
    require_both_endpoints: bool = False,
) -> np.ndarray:
    """Conservative hole filling using slice-constrained approach.

    Uses fill_holes_slice_constrained which only fills small holes
    that are well-surrounded by foreground.

    This is the most conservative option, least likely to cause bridging.

    Args:
        pr: Probability map
        threshold: Initial threshold
        min_size: Minimum component size
        border_standoff: Border distance for filtering
        require_both_endpoints: Require both endpoints near border

    """
    if border_standoff is None:
        border_standoff = GT_PARAMS["border_standoff"]

    mask = threshold_gt_optimized(pr, threshold)

    # Conservative slice-by-slice hole filling
    mask = fill_holes_slice_constrained(
        mask,
        max_hole_size=30,  # Very small holes only
        max_hole_ratio=0.3,  # Must be mostly surrounded
    )

    mask = remove_small_gt(mask, min_size=min_size)

    min_endpoints = 2 if require_both_endpoints else 1
    mask = filter_by_border_endpoints(
        mask,
        border_standoff=border_standoff,
        require_start_near_border=require_both_endpoints,
        require_end_near_border=require_both_endpoints,
        min_endpoint_near_border=min_endpoints,
    )

    return mask


def postprocess_border_aware_hysteresis(
    pr: np.ndarray,
    low_threshold: float = 0.10,
    high_threshold: float = 0.25,
    min_size: int = 500,
    min_z_span: int = 30,
    border_standoff: int = None,
) -> np.ndarray:
    """Hysteresis thresholding with border-aware filtering.

    Combines robust hysteresis thresholding with border endpoint filtering.
    """
    if border_standoff is None:
        border_standoff = GT_PARAMS["border_standoff"]

    # Hysteresis thresholding
    mask = hysteresis_gt_informed(pr, low_threshold, high_threshold)

    # Basic cleanup
    mask = fill_holes_gt(mask, fill_2d=True, fill_3d=True)
    mask = remove_small_gt(mask, min_size=min_size)

    # Z-span filtering (ribbons should span significant z-range)
    mask = remove_fragments_by_z_span(mask, min_z_span=min_z_span)

    # Border filtering - keep ribbons with at least one endpoint near border
    mask = filter_by_border_endpoints(
        mask,
        border_standoff=border_standoff,
        require_start_near_border=False,
        require_end_near_border=False,
        min_endpoint_near_border=1,  # At least one endpoint near border
    )

    return mask


def postprocess_border_strict(
    pr: np.ndarray,
    low_threshold: float = 0.12,
    high_threshold: float = 0.28,
    min_size: int = 1000,
    min_z_span: int = 40,
    border_standoff: int = None,
) -> np.ndarray:
    """Strict border-aware filtering requiring BOTH endpoints near borders.

    This is the most aggressive border filter - only keeps ribbons that
    clearly follow the GT pattern of starting AND ending near XY borders.
    """
    if border_standoff is None:
        border_standoff = GT_PARAMS["border_standoff"]

    # Hysteresis thresholding
    mask = hysteresis_gt_informed(pr, low_threshold, high_threshold)

    # Full cleanup pipeline
    mask = fill_holes_gt(mask, fill_2d=True, fill_3d=True)
    mask = morphology_gt(mask, closing_size=1, opening_size=1)
    mask = remove_topology_defects(mask)
    mask = remove_small_gt(mask, min_size=min_size)
    mask = remove_fragments_by_z_span(mask, min_z_span=min_z_span)

    # Strict border filtering - BOTH endpoints must be near border
    mask = filter_by_border_endpoints(
        mask,
        border_standoff=border_standoff,
        require_start_near_border=True,
        require_end_near_border=True,
        min_endpoint_near_border=2,
    )

    return mask


def postprocess_border_path_aware(
    pr: np.ndarray,
    threshold: float = 0.18,
    min_size: int = 500,
    min_z_span: int = 30,
    border_standoff: int = None,
    max_displacement: float = None,
) -> np.ndarray:
    """Combined border and path smoothness filtering.

    Uses both border connectivity and path smoothness priors to filter
    out components that don't match expected ribbon characteristics.
    """
    if border_standoff is None:
        border_standoff = GT_PARAMS["border_standoff"]
    if max_displacement is None:
        max_displacement = GT_PARAMS["max_centroid_displacement"]

    # Initial processing
    mask = threshold_gt_optimized(pr, threshold)
    mask = fill_holes_gt(mask, fill_2d=True, fill_3d=True)
    mask = remove_small_gt(mask, min_size=min_size)
    mask = remove_fragments_by_z_span(mask, min_z_span=min_z_span)

    # Filter by path smoothness (removes erratic/noisy components)
    mask = filter_by_path_smoothness(
        mask,
        max_displacement_per_slice=max_displacement,
        min_path_smoothness=0.25,  # Lenient smoothness threshold
    )

    # Filter by border endpoints (at least one endpoint near border)
    mask = filter_by_border_endpoints(
        mask,
        border_standoff=border_standoff,
        require_start_near_border=False,
        require_end_near_border=False,
        min_endpoint_near_border=1,
    )

    return mask


def postprocess_border_extend(
    pr: np.ndarray,
    threshold: float = 0.15,
    min_size: int = 300,
    border_standoff: int = None,
    extension_threshold: float = 0.08,
) -> np.ndarray:
    """Border-aware processing with ribbon extension toward borders.

    This method attempts to extend partial ribbons toward borders
    using probability guidance, then filters by border connectivity.

    Useful when model predictions are truncated before reaching borders.
    """
    if border_standoff is None:
        border_standoff = GT_PARAMS["border_standoff"]

    # Lower threshold to capture more of the ribbon
    mask = threshold_gt_optimized(pr, threshold)
    mask = fill_holes_gt(mask, fill_2d=True, fill_3d=True)
    mask = remove_small_gt(mask, min_size=min_size)

    # Try to extend ribbons toward borders
    mask = extend_ribbons_to_borders(
        mask,
        pr,
        border_standoff=border_standoff,
        extension_threshold=extension_threshold,
    )

    # Now filter by border connectivity
    mask = filter_by_border_endpoints(
        mask,
        border_standoff=border_standoff,
        require_start_near_border=False,
        require_end_near_border=False,
        min_endpoint_near_border=1,
    )

    # Final cleanup
    mask = remove_small_gt(mask, min_size=min_size)

    return mask


def postprocess_full_pipeline(
    pr: np.ndarray,
    low_threshold: float = 0.10,
    high_threshold: float = 0.25,
    min_size: int = 800,
    min_z_span: int = 35,
    border_standoff: int = None,
    use_border_filter: bool = True,
    use_path_filter: bool = True,
    use_morphology: bool = True,
) -> np.ndarray:
    """Full pipeline combining ALL GT priors for maximum accuracy.

    This is the most comprehensive pipeline using:
    1. Hysteresis thresholding
    2. Hole filling
    3. Morphological cleanup
    4. Topology defect removal
    5. Size and z-span filtering
    6. Border endpoint filtering (99.6% prior)
    7. Path smoothness filtering

    Args:
        pr: Probability map
        low_threshold: Low threshold for hysteresis
        high_threshold: High threshold for hysteresis
        min_size: Minimum component size
        min_z_span: Minimum z-span
        border_standoff: Border distance threshold
        use_border_filter: Apply border endpoint filtering
        use_path_filter: Apply path smoothness filtering
        use_morphology: Apply morphological operations

    """
    if border_standoff is None:
        border_standoff = GT_PARAMS["border_standoff"]

    # 1. Hysteresis thresholding for robust initial segmentation
    mask = hysteresis_gt_informed(pr, low_threshold, high_threshold)

    # 2. Fill holes (GT: 2.3% have holes)
    mask = fill_holes_gt(mask, fill_2d=True, fill_3d=True)

    # 3. Morphological cleanup
    if use_morphology:
        mask = morphology_gt(mask, closing_size=1, opening_size=1)

    # 4. Remove topology defects
    mask = remove_topology_defects(mask, remove_tunnels=True, remove_voids=True)

    # 5. Size filtering
    mask = remove_small_gt(mask, min_size=min_size)

    # 6. Z-span filtering
    mask = remove_fragments_by_z_span(mask, min_z_span=min_z_span)

    # 7. Z-continuity enforcement
    mask = enforce_z_continuity_gt(mask, min_z_span=min_z_span, min_overlap_ratio=0.15)

    # 8. Path smoothness filtering (removes erratic components)
    if use_path_filter:
        mask = filter_by_path_smoothness(
            mask,
            max_displacement_per_slice=GT_PARAMS["max_centroid_displacement"] * 1.5,
            min_path_smoothness=0.2,
        )

    # 9. Border endpoint filtering (the key prior!)
    if use_border_filter:
        mask = filter_by_border_endpoints(
            mask,
            border_standoff=border_standoff,
            require_start_near_border=False,
            require_end_near_border=False,
            min_endpoint_near_border=1,
        )

    return mask


def postprocess_adaptive_threshold(
    pr: np.ndarray,
    base_threshold: float = 0.18,
    border_standoff: int = None,
    min_size: int = 500,
) -> np.ndarray:
    """Adaptive thresholding based on local probability distribution.

    Uses different thresholds for border regions vs interior,
    since ribbons should have higher probability near borders.
    """
    if border_standoff is None:
        border_standoff = GT_PARAMS["border_standoff"]

    z_depth, h, w = pr.shape

    # Create border mask
    border_mask = np.zeros((h, w), dtype=bool)
    border_mask[:border_standoff, :] = True
    border_mask[-border_standoff:, :] = True
    border_mask[:, :border_standoff] = True
    border_mask[:, -border_standoff:] = True

    # Expand to 3D
    border_mask_3d = np.broadcast_to(border_mask[np.newaxis, :, :], pr.shape)

    # Lower threshold near borders to capture ribbon endpoints
    threshold_map = np.where(border_mask_3d, base_threshold * 0.7, base_threshold)

    # Apply adaptive threshold
    mask = (pr > threshold_map).astype(np.uint8)

    # Cleanup
    mask = fill_holes_gt(mask, fill_2d=True, fill_3d=True)
    mask = remove_small_gt(mask, min_size=min_size)

    # Border filtering
    mask = filter_by_border_endpoints(
        mask,
        border_standoff=border_standoff,
        min_endpoint_near_border=1,
    )

    return mask


# =============================================================================
# MAIN POSTPROCESS FUNCTION FOR INTEGRATION
# =============================================================================


def postprocess_gt(
    pr: np.ndarray,
    method: str = "balanced",
    params: dict = None,
) -> np.ndarray:
    """Main entry point for GT-informed post-processing.

    Args:
        pr: Probability map (0-1)
        method: Post-processing method name
        params: Optional parameter overrides

    Returns:
        Binary mask

    Available methods:
        Original methods:
        - threshold_gt: Simple threshold
        - hysteresis_gt: Hysteresis thresholding
        - gt_minimal: Minimal processing (threshold + hole fill)
        - gt_balanced: Balanced processing
        - gt_topology: Topology-focused
        - gt_strict: Strict filtering
        - gt_surface: Surface dice optimized
        - gt_voi: VOI optimized
        - gt_comprehensive: Full pipeline

        Border-aware methods (leverage 99.6% border connectivity prior):
        - gt_border_aware: Basic border endpoint filtering
        - gt_border_hysteresis: Hysteresis + border filtering
        - gt_border_strict: Strict border filtering (both endpoints)
        - gt_border_path: Combined border + path smoothness filtering
        - gt_border_extend: Extend ribbons toward borders
        - gt_full_pipeline: All GT priors combined
        - gt_adaptive: Adaptive thresholding with border awareness

        Safe hole-filling methods (for thin ribbons that are close together):
        - gt_hole_fill_safe: Component-aware safe hole filling with distance constraints
        - gt_hole_fill_conservative: Conservative slice-constrained hole filling

        NEW CC3D-based methods (advanced topology processing):
        - cc3d_topology: Basic cc3d topology pipeline
        - cc3d_advanced: Advanced cc3d pipeline with shape/contact analysis
        - cc3d_iterative: Multi-level iterative refinement
        - cc3d_topology_strict: Strict topology-focused pipeline
        - cc3d_largest: Keep K largest components
        - cc3d_full: Full cc3d pipeline with all features
        - cc3d_shape: Shape-based filtering only
        - cc3d_statistics: Statistics-based filtering
        - cc3d_continuous: Continuous value CCL
        - cc3d_watershed: Watershed-based separation

    """
    if params is None:
        params = {}

    methods = {
        # Original methods
        "threshold_gt": lambda: threshold_gt_optimized(pr, **params),
        "hysteresis_gt": lambda: hysteresis_gt_informed(pr, **params),
        "gt_minimal": lambda: postprocess_gt_minimal(pr, **params),
        "gt_balanced": lambda: postprocess_gt_balanced(pr, **params),
        "gt_topology": lambda: postprocess_gt_topology(pr, **params),
        "gt_strict": lambda: postprocess_gt_strict(pr, **params),
        "gt_surface": lambda: postprocess_gt_surface_optimized(pr, **params),
        "gt_voi": lambda: postprocess_gt_voi_optimized(pr, **params),
        "gt_comprehensive": lambda: postprocess_gt_comprehensive(pr, **params),
        # Border-aware methods
        "gt_border_aware": lambda: postprocess_border_aware(pr, **params),
        "gt_border_hysteresis": lambda: postprocess_border_aware_hysteresis(pr, **params),
        "gt_border_strict": lambda: postprocess_border_strict(pr, **params),
        "gt_border_path": lambda: postprocess_border_path_aware(pr, **params),
        "gt_border_extend": lambda: postprocess_border_extend(pr, **params),
        "gt_full_pipeline": lambda: postprocess_full_pipeline(pr, **params),
        "gt_adaptive": lambda: postprocess_adaptive_threshold(pr, **params),
        # NEW: Safe hole-filling methods (for thin ribbons near each other)
        "gt_hole_fill_safe": lambda: postprocess_hole_fill_safe(pr, **params),
        "gt_hole_fill_conservative": lambda: postprocess_hole_fill_conservative(pr, **params),
        # NEW: CC3D-based methods
        "cc3d_topology": lambda: postprocess_cc3d_topology(pr, **params),
        "cc3d_advanced": lambda: postprocess_cc3d_advanced(pr, **params),
        "cc3d_iterative": lambda: postprocess_cc3d_iterative(pr, **params),
        "cc3d_topology_strict": lambda: postprocess_cc3d_topology_strict(pr, **params),
        "cc3d_largest": lambda: postprocess_cc3d_largest(pr, **params),
        "cc3d_full": lambda: postprocess_cc3d_full(pr, **params),
        "cc3d_shape": lambda: cc3d_shape_filter(
            (pr > params.get("threshold", 0.18)).astype(np.uint8),
            **{k: v for k, v in params.items() if k != "threshold"},
        ),
        "cc3d_statistics": lambda: cc3d_statistics_filter(
            (pr > params.get("threshold", 0.18)).astype(np.uint8),
            **{k: v for k, v in params.items() if k != "threshold"},
        ),
        "cc3d_continuous": lambda: cc3d_continuous_threshold(pr, **params),
        "cc3d_watershed": lambda: cc3d_watershed_separation(pr, **params),
    }

    if method not in methods:
        raise ValueError(f"Unknown method: {method}. Available: {list(methods.keys())}")

    return methods[method]()


if __name__ == "__main__":
    print("GT-Informed Post-Processing Functions")
    print("=" * 50)
    print("\nGT Parameters:")
    for key, value in GT_PARAMS.items():
        print(f"  {key}: {value}")

    print("\nAvailable methods:")
    print("\n  Original methods:")
    original_methods = [
        "threshold_gt",
        "hysteresis_gt",
        "gt_minimal",
        "gt_balanced",
        "gt_topology",
        "gt_strict",
        "gt_surface",
        "gt_voi",
        "gt_comprehensive",
    ]
    for m in original_methods:
        print(f"    - {m}")

    print("\n  Border-aware methods (99.6% border connectivity prior):")
    border_methods = [
        "gt_border_aware",
        "gt_border_hysteresis",
        "gt_border_strict",
        "gt_border_path",
        "gt_border_extend",
        "gt_full_pipeline",
        "gt_adaptive",
    ]
    for m in border_methods:
        print(f"    - {m}")

    print("\n  NEW CC3D-based methods (advanced topology processing):")
    cc3d_methods = [
        "cc3d_topology",
        "cc3d_advanced",
        "cc3d_iterative",
        "cc3d_topology_strict",
        "cc3d_largest",
        "cc3d_full",
        "cc3d_shape",
        "cc3d_statistics",
        "cc3d_continuous",
        "cc3d_watershed",
    ]
    for m in cc3d_methods:
        print(f"    - {m}")
