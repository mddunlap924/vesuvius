"""Precision-focused post-processing strategies for thin ribbon segmentation.

These methods target the key bottleneck identified in low-scoring samples:
excessive false positive foreground (precision ~10-45%) that creates extra
connected components and tanks both topo_dim0 and VOI scores.

Key insight from analysis of 30 samples:
- Low scorers (<0.6): precision ~10-45%, recall ~70-90%
- High scorers (>0.75): precision ~60-80%, recall ~80-95%
- Surface dice is already good (0.88-0.99)
- Topo is the #1 bottleneck (0.27 vs 0.94)

Strategy: Aggressively remove false positive regions while preserving true ribbons.

Created: 2026-02-05
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import cc3d
import numpy as np
from scipy import ndimage
from scipy.ndimage import label as scipy_label
from scipy.stats import kurtosis

logger = logging.getLogger(__name__)


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


def _remove_small_components(mask: np.ndarray, min_size: int = 500) -> np.ndarray:
    """Remove connected components smaller than min_size."""
    labels = cc3d.connected_components(mask, connectivity=26)
    stats = cc3d.statistics(labels)
    for i in range(1, labels.max() + 1):
        if stats["voxel_counts"][i] < min_size:
            mask[labels == i] = 0
    return mask


def _remove_small_and_flat(
    mask: np.ndarray,
    min_size: int = 3000,
    min_z_span: int = 15,
) -> np.ndarray:
    """Remove components that are too small or don't span enough z-slices."""
    labels = cc3d.connected_components(mask, connectivity=26)
    stats = cc3d.statistics(labels)
    bb = stats["bounding_boxes"]
    for i in range(1, labels.max() + 1):
        z_span = bb[i][0].stop - bb[i][0].start
        if stats["voxel_counts"][i] < min_size or z_span < min_z_span:
            mask[labels == i] = 0
    return mask


def _border_endpoint_filter(
    mask: np.ndarray,
    border_standoff: int = 15,
    min_endpoints: int = 1,
) -> np.ndarray:
    """Keep only components whose endpoints are near XY borders.

    GT analysis: 99.6% of ribbons start/end near borders.
    """
    labeled, n = scipy_label(mask)
    result = np.zeros_like(mask)

    for i in range(1, n + 1):
        comp = labeled == i
        # Get z-range for this component
        z_present = np.any(np.any(comp, axis=2), axis=1)
        z_where = np.where(z_present)[0]
        if len(z_where) < 2:
            continue

        first_z, last_z = z_where[0], z_where[-1]
        n_near = 0

        for z in [first_z, last_z]:
            slice_2d = comp[z]
            ys, xs = np.where(slice_2d)
            if len(ys) == 0:
                continue
            h, w = slice_2d.shape
            near = (
                ys.min() < border_standoff
                or ys.max() >= h - border_standoff
                or xs.min() < border_standoff
                or xs.max() >= w - border_standoff
            )
            if near:
                n_near += 1

        if n_near >= min_endpoints:
            result[comp] = 1

    return result


def _estimate_fg_fraction(pr: np.ndarray, threshold: float) -> float:
    """Estimate foreground fraction at a given threshold."""
    return float((pr > threshold).mean())


def _find_threshold_for_target_fg(
    pr: np.ndarray,
    target_fg: float,
    low: float = 0.05,
    high: float = 0.95,
    n_iter: int = 25,
) -> float:
    """Binary search for threshold that achieves target foreground fraction."""
    for _ in range(n_iter):
        mid = (low + high) / 2.0
        fg = _estimate_fg_fraction(pr, mid)
        if fg > target_fg:
            low = mid
        else:
            high = mid
    return (low + high) / 2.0


# =============================================================================
# STRATEGY 1: PRECISION-ADAPTIVE THRESHOLDING
# =============================================================================


def precision_adaptive_postprocess(
    pr: np.ndarray,
    base_threshold: float = 0.20,
    min_size: int = 3000,
    min_z_span: int = 15,
    apply_border_filter: bool = True,
    border_standoff: int = 15,
    target_fg_fraction: float = 0.10,
    max_fg_ratio: float = 2.5,
) -> np.ndarray:
    """Precision-adaptive thresholding based on predicted FG volume.

    Key insight: Low-scoring samples have massive false positive regions.
    If predicted FG is much larger than expected GT volume (~5-15%),
    we raise the threshold to reduce FP while preserving true ribbons.

    The strategy:
    1. Start with base threshold
    2. If FG fraction exceeds target by more than max_fg_ratio, raise threshold
    3. Apply aggressive structural filtering (z-span + border)

    Args:
        pr: Probability map (float32, 0-1)
        base_threshold: Starting threshold (from adaptive method)
        min_size: Minimum component size
        min_z_span: Minimum z-span for a valid component
        apply_border_filter: Whether to apply border endpoint filtering
        border_standoff: Distance from border for endpoint check
        target_fg_fraction: Expected GT foreground fraction
        max_fg_ratio: Max ratio of pred FG / target FG before raising threshold

    Returns:
        Binary mask (uint8)

    """
    # Compute FG fraction at base threshold
    fg_frac = _estimate_fg_fraction(pr, base_threshold)

    # If FG is within acceptable range, use base threshold
    if fg_frac <= target_fg_fraction * max_fg_ratio:
        threshold = base_threshold
    else:
        # FG is too large — find threshold that gives acceptable FG fraction
        # Use a slightly generous target to avoid being too aggressive
        adjusted_target = target_fg_fraction * 1.5
        threshold = _find_threshold_for_target_fg(pr, adjusted_target)
        # Don't go below base threshold (that would increase FG)
        threshold = max(threshold, base_threshold)
        logger.info(
            f"precision_adaptive: fg={fg_frac:.3f} too high, "
            f"raised threshold {base_threshold:.2f} -> {threshold:.3f}",
        )

    # Apply threshold
    mask = (pr > threshold).astype(np.uint8)

    # Aggressive structural filtering
    mask = _remove_small_and_flat(mask, min_size=min_size, min_z_span=min_z_span)

    # Border endpoint filter
    if apply_border_filter:
        mask = _border_endpoint_filter(mask, border_standoff=border_standoff)

    return mask


# =============================================================================
# STRATEGY 2: MORPHOLOGICAL OPENING CLEANUP
# =============================================================================


def opening_cleanup_postprocess(
    pr: np.ndarray,
    threshold: float = 0.20,
    opening_radius: int = 2,
    min_size: int = 2000,
    min_z_span: int = 10,
    apply_border_filter: bool = True,
    border_standoff: int = 15,
) -> np.ndarray:
    """Morphological opening to remove thin FP noise while preserving thick ribbons.

    Opening = erosion followed by dilation. This removes thin structures
    (noise, FP bridges) while preserving thick ribbon-like structures.

    Key insight: True ribbons are thick sheet-like structures that survive
    erosion, while FP noise is often thin and fragmented.

    Args:
        pr: Probability map
        threshold: Initial threshold
        opening_radius: Number of erosion+dilation iterations
        min_size: Minimum component size after opening
        min_z_span: Minimum z-span
        apply_border_filter: Whether to apply border filtering
        border_standoff: Border distance

    Returns:
        Binary mask (uint8)

    """
    mask = (pr > threshold).astype(np.uint8)

    # Morphological opening with 6-connectivity (conservative)
    struct = ndimage.generate_binary_structure(3, 1)
    eroded = ndimage.binary_erosion(mask, structure=struct, iterations=opening_radius)
    opened = ndimage.binary_dilation(
        eroded,
        structure=struct,
        iterations=opening_radius,
    ).astype(np.uint8)

    # Remove small and flat components
    opened = _remove_small_and_flat(opened, min_size=min_size, min_z_span=min_z_span)

    if apply_border_filter:
        opened = _border_endpoint_filter(opened, border_standoff=border_standoff)

    return opened


# =============================================================================
# STRATEGY 3: VOLUME-CALIBRATED THRESHOLDING
# =============================================================================


def volume_calibrated_postprocess(
    pr: np.ndarray,
    target_fg_fraction: float = 0.08,
    min_size: int = 3000,
    min_z_span: int = 15,
    apply_border_filter: bool = True,
    border_standoff: int = 15,
) -> np.ndarray:
    """Calibrate threshold to match expected GT volume fraction.

    Instead of a fixed threshold, find the threshold that gives
    a specific foreground fraction matching typical GT volumes.

    GT analysis shows ribbons typically occupy 5-15% of volume.

    Args:
        pr: Probability map
        target_fg_fraction: Target FG fraction (default: 0.08 from GT analysis)
        min_size: Minimum component size
        min_z_span: Minimum z-span
        apply_border_filter: Apply border filter
        border_standoff: Border standoff

    Returns:
        Binary mask (uint8)

    """
    threshold = _find_threshold_for_target_fg(pr, target_fg_fraction)
    logger.info(
        f"volume_calibrated: target_fg={target_fg_fraction:.3f} -> threshold={threshold:.3f}"
    )

    mask = (pr > threshold).astype(np.uint8)
    mask = _remove_small_and_flat(mask, min_size=min_size, min_z_span=min_z_span)

    if apply_border_filter:
        mask = _border_endpoint_filter(mask, border_standoff=border_standoff)

    return mask


# =============================================================================
# STRATEGY 4: HIGH-CONFIDENCE SEED + CONTROLLED GROWTH
# =============================================================================


def highconf_seed_postprocess(
    pr: np.ndarray,
    seed_threshold: float = 0.50,
    grow_threshold: float = 0.15,
    min_size: int = 3000,
    min_z_span: int = 15,
    apply_border_filter: bool = True,
    border_standoff: int = 15,
) -> np.ndarray:
    """Start from high-confidence seeds and grow into nearby probable regions.

    1. Create seeds from high-confidence voxels
    2. Grow seeds into connected regions above a lower threshold
    3. This gives better precision than low-threshold alone, better recall than high-threshold

    Args:
        pr: Probability map
        seed_threshold: High threshold for seed creation
        grow_threshold: Low threshold for growth region
        min_size: Minimum component size
        min_z_span: Minimum z-span
        apply_border_filter: Apply border filtering
        border_standoff: Border standoff

    Returns:
        Binary mask (uint8)

    """
    # Seed mask: high-confidence voxels
    seeds = pr > seed_threshold

    # Growth region: lower-confidence voxels
    growth_region = pr > grow_threshold

    # Label connected components in growth region
    labeled, num_features = scipy_label(growth_region)

    # Find which growth components contain seeds
    seed_labels = np.unique(labeled[seeds])
    seed_labels = seed_labels[seed_labels != 0]

    # Keep only growth components that have seeds
    mask = np.isin(labeled, seed_labels).astype(np.uint8)

    # Structural filtering
    mask = _remove_small_and_flat(mask, min_size=min_size, min_z_span=min_z_span)

    if apply_border_filter:
        mask = _border_endpoint_filter(mask, border_standoff=border_standoff)

    return mask


# =============================================================================
# STRATEGY 5: COMBINED PRECISION PIPELINE
# =============================================================================


def combined_precision_postprocess(
    pr: np.ndarray,
    min_size: int = 3000,
    min_z_span: int = 15,
    border_standoff: int = 15,
) -> np.ndarray:
    """Combined precision pipeline - best strategies merged.

    Adapts the strategy based on the probability distribution:
    - High FP samples (fg > 25% at t=0.20): Use precision-adaptive with volume calibration
    - Moderate FP (fg 15-25%): Use high-conf seed approach
    - Low FP (fg < 15%): Use standard adaptive + border (current best)

    This is the main entry point for the precision-focused approach.

    Args:
        pr: Probability map
        min_size: Minimum component size
        min_z_span: Minimum z-span
        border_standoff: Border standoff

    Returns:
        Binary mask (uint8)

    """
    fg_at_020 = _estimate_fg_fraction(pr, 0.20)

    if fg_at_020 > 0.25:
        # HIGH FP: Very aggressive precision approach
        # These samples have 2.5-8x too much foreground
        logger.info(f"combined_precision: HIGH FP (fg={fg_at_020:.3f}), using volume calibration")
        mask = volume_calibrated_postprocess(
            pr,
            target_fg_fraction=0.10,
            min_size=min_size,
            min_z_span=min_z_span,
            apply_border_filter=True,
            border_standoff=border_standoff,
        )
    elif fg_at_020 > 0.15:
        # MODERATE FP: Use high-conf seed approach
        logger.info(f"combined_precision: MODERATE FP (fg={fg_at_020:.3f}), using highconf seeds")
        mask = highconf_seed_postprocess(
            pr,
            seed_threshold=0.45,
            grow_threshold=0.15,
            min_size=min_size,
            min_z_span=min_z_span,
            apply_border_filter=True,
            border_standoff=border_standoff,
        )
    else:
        # LOW FP: Standard approach is already good
        # Just add z-span and border filtering
        logger.info(f"combined_precision: LOW FP (fg={fg_at_020:.3f}), using standard adaptive")

        # Get adaptive threshold
        flat = pr.flatten()
        pk = kurtosis(flat)
        pm = np.mean(flat)

        if pk > 4.5 and pm < 0.09:
            threshold = 0.10
        elif pk > 4.0 and pm < 0.10:
            threshold = 0.15
        else:
            threshold = 0.20

        mask = (pr > threshold).astype(np.uint8)
        mask = _remove_small_and_flat(mask, min_size=min_size, min_z_span=min_z_span)
        mask = _border_endpoint_filter(mask, border_standoff=border_standoff)

    return mask


# =============================================================================
# STRATEGY 6: PRECISION-ADAPTIVE BORDER-AWARE (ENHANCED)
# =============================================================================


def precision_border_aware_postprocess(
    pr: np.ndarray,
    min_size: int = 500,
    border_standoff: int = 15,
) -> np.ndarray:
    """Enhanced adaptive_border_aware with precision-based corrections.

    This enhances the current best method (adaptive_border_aware) by:
    1. Using the same adaptive threshold
    2. Adding z-span filtering (removes 2D noise artifacts)
    3. Adding precision-based threshold adjustment for high-FP samples
    4. Using morphological opening for samples with excessive FP

    The goal is to improve low-scoring samples without regressing high ones.

    Args:
        pr: Probability map
        min_size: Minimum component size
        border_standoff: Border standoff distance

    Returns:
        Binary mask (uint8)

    """
    flat = pr.flatten()
    pk = kurtosis(flat)
    pm = np.mean(flat)

    # Compute adaptive threshold (same as current method)
    if pk > 4.5 and pm < 0.09:
        threshold = 0.10
    elif pk > 4.0 and pm < 0.10:
        threshold = 0.15
    else:
        threshold = 0.20

    # Check FP level at this threshold
    fg_frac = _estimate_fg_fraction(pr, threshold)

    # For high-FP samples, apply additional precision measures
    if fg_frac > 0.20:
        # Heavy FP: raise threshold and/or apply opening
        # Find threshold giving ~15% FG (typical upper bound for GT)
        adjusted_thresh = _find_threshold_for_target_fg(pr, 0.15)
        threshold = max(threshold, adjusted_thresh)
        logger.info(
            f"precision_border_aware: fg={fg_frac:.3f} too high, "
            f"adjusted threshold to {threshold:.3f}",
        )

        mask = (pr > threshold).astype(np.uint8)

        # Light morphological opening to clean thin FP noise
        struct = ndimage.generate_binary_structure(3, 1)
        eroded = ndimage.binary_erosion(mask, structure=struct, iterations=1)
        mask = ndimage.binary_dilation(eroded, structure=struct, iterations=1).astype(np.uint8)
    else:
        # Normal FP: standard threshold
        mask = (pr > threshold).astype(np.uint8)

    # Remove small components
    mask = _remove_small_components(mask, min_size=min_size)

    # Z-span filtering (remove flat/2D artifacts)
    labels = cc3d.connected_components(mask, connectivity=26)
    stats = cc3d.statistics(labels)
    bb = stats["bounding_boxes"]
    for i in range(1, labels.max() + 1):
        z_span = bb[i][0].stop - bb[i][0].start
        if z_span < 10:
            mask[labels == i] = 0

    # Border endpoint filtering
    mask = _border_endpoint_filter(mask, border_standoff=border_standoff)

    return mask


# =============================================================================
# STRATEGY 7: MULTI-SCALE CONSENSUS
# =============================================================================


def multiscale_consensus_postprocess(
    pr: np.ndarray,
    thresholds: tuple[float, ...] = (0.15, 0.20, 0.25, 0.30, 0.35),
    min_agreement: int = 3,
    min_size: int = 3000,
    min_z_span: int = 15,
    apply_border_filter: bool = True,
    border_standoff: int = 15,
) -> np.ndarray:
    """Multi-threshold consensus: keep voxels that are foreground at multiple thresholds.

    Idea: True ribbon voxels have high probability and survive many thresholds.
    FP noise voxels have marginal probability and only survive low thresholds.
    By requiring agreement across multiple thresholds, we suppress FP.

    Args:
        pr: Probability map
        thresholds: Tuple of thresholds to test
        min_agreement: Minimum number of thresholds that must agree
        min_size: Minimum component size
        min_z_span: Minimum z-span
        apply_border_filter: Apply border filter
        border_standoff: Border standoff

    Returns:
        Binary mask (uint8)

    """
    # Count how many thresholds each voxel survives
    vote_map = np.zeros_like(pr, dtype=np.uint8)
    for t in thresholds:
        vote_map += (pr > t).astype(np.uint8)

    # Keep voxels with sufficient agreement
    mask = (vote_map >= min_agreement).astype(np.uint8)

    # Structural filtering
    mask = _remove_small_and_flat(mask, min_size=min_size, min_z_span=min_z_span)

    if apply_border_filter:
        mask = _border_endpoint_filter(mask, border_standoff=border_standoff)

    return mask
