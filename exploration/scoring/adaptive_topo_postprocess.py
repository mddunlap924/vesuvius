"""Adaptive Topology-Focused Post-Processing Module (v3).

This module provides sample-specific post-processing strategies optimized for
topological score (betti-matching) improvement. The strategies are based on
analysis of 80 samples with known ground truth, identifying which probability
distribution characteristics correlate with different post-processing needs.

Key Insights from 80-Sample Analysis (2026-02-02/03):
=====================================================

1. TOPO SCORE CORRELATIONS (Spearman):
   - decisive_ratio:       r=+0.51 (p<0.001) **strongest predictor**
   - bimodality_coef:      r=+0.47 (p<0.001)
   - frac_low_conf:        r=-0.43 (p<0.001)
   - num_components:       r=-0.43 (p<0.001)
   - frac_medium_conf:     r=-0.44 (p<0.001)
   - prob_skewness:        r=+0.30 (p<0.01)
   - prob_kurtosis:        r=+0.27 (p<0.05)

2. PROBLEM CATEGORIES IDENTIFIED (updated 2026-02-03):
   a) HIGH FRAGMENTATION (>100 components at thresh 0.20): 23 samples
      - Mean topo=0.31 vs overall 0.44
      - Mean decisive_ratio: 0.51
      - SOLUTION: Keep-K largest + aggressive min_size + closing

   b) LOW DECISIVE RATIO (<0.40): 8 samples
      - Mean topo=0.17 (worst category!)
      - Mean components: 223
      - SOLUTION: Hysteresis with strict high seeds, extend to low-conf regions

   c) HIGH CONFIDENCE (decisive_ratio > 0.60): 46 samples
      - Mean topo=0.52 (best category)
      - Mean components: 64
      - SOLUTION: Conservative - lower threshold (0.10-0.15), minimal cleanup

3. BEST SAMPLES CHARACTERISTICS (topo > 0.7): 11 samples
   - decisive_ratio: 0.68 (vs 0.51 overall)
   - num_components: 50 (vs 80 overall)
   - prob_kurtosis: 4.75 (vs 4.08 overall)

4. WORST SAMPLES CHARACTERISTICS (topo < 0.2): 11 samples
   - decisive_ratio: 0.46 (uncertain)
   - num_components: 171 (fragmented)

Usage:
======
    from adaptive_topo_postprocess import adaptive_topo_postprocess

    # Apply sample-specific post-processing
    binary_mask = adaptive_topo_postprocess(probs, strategy="auto")

    # Get just the features for logging/analysis
    features = extract_topo_features(probs)

    # Get recommended strategy without applying
    strategy, params = recommend_strategy(probs)
"""

import warnings
from typing import NamedTuple

import cc3d
import numpy as np
from scipy import ndimage
from scipy.ndimage import (
    binary_closing,
    binary_dilation,
    binary_erosion,
    binary_fill_holes,
    generate_binary_structure,
)
from scipy.ndimage import label as scipy_label
from scipy.stats import kurtosis, skew

warnings.filterwarnings("ignore")


# =============================================================================
# FEATURE EXTRACTION
# =============================================================================


class TopoFeatures(NamedTuple):
    """Key features for topology-focused post-processing decisions."""

    prob_mean: float
    prob_std: float
    prob_kurtosis: float
    prob_skewness: float
    decisive_ratio: float
    bimodality_coef: float
    frac_high_conf: float
    frac_medium_conf: float
    frac_low_conf: float
    num_components_t10: int
    num_components_t15: int
    num_components_t20: int
    volume_concentration_t20: float


def extract_topo_features(probs: np.ndarray) -> TopoFeatures:
    """Extract features relevant for topology-focused post-processing.

    Args:
        probs: 3D probability array (D, H, W) with values 0-1

    Returns:
        TopoFeatures named tuple with all relevant metrics

    """
    # Handle multi-channel input
    if probs.ndim == 4 and probs.shape[0] == 2:
        probs = probs[1]  # Foreground channel

    flat_probs = probs.flatten()

    # Basic stats
    prob_mean = np.mean(flat_probs)
    prob_std = np.std(flat_probs)
    prob_kurtosis_val = kurtosis(flat_probs)
    prob_skewness_val = skew(flat_probs)

    # Confidence regions
    frac_high_conf = (flat_probs > 0.7).mean()
    frac_medium_conf = ((flat_probs > 0.3) & (flat_probs <= 0.7)).mean()
    frac_low_conf = ((flat_probs > 0.1) & (flat_probs <= 0.3)).mean()

    # Decisive ratio: fraction of positive predictions that are high confidence
    positive_mask = flat_probs > 0.15
    if positive_mask.sum() > 0:
        decisive_ratio = (flat_probs[positive_mask] > 0.7).mean()
    else:
        decisive_ratio = 0.0

    # Bimodality coefficient: (skewness^2 + 1) / (kurtosis + 3)
    bimodality_coef = (
        (prob_skewness_val**2 + 1) / (prob_kurtosis_val + 3) if (prob_kurtosis_val + 3) > 0 else 0.5
    )

    # Component counts at different thresholds
    num_components_t10 = _count_components(probs > 0.10)
    num_components_t15 = _count_components(probs > 0.15)
    num_components_t20 = _count_components(probs > 0.20)

    # Volume concentration at t20 (largest component / total volume)
    volume_concentration_t20 = _compute_volume_concentration(probs > 0.20)

    return TopoFeatures(
        prob_mean=prob_mean,
        prob_std=prob_std,
        prob_kurtosis=prob_kurtosis_val,
        prob_skewness=prob_skewness_val,
        decisive_ratio=decisive_ratio,
        bimodality_coef=bimodality_coef,
        frac_high_conf=frac_high_conf,
        frac_medium_conf=frac_medium_conf,
        frac_low_conf=frac_low_conf,
        num_components_t10=num_components_t10,
        num_components_t15=num_components_t15,
        num_components_t20=num_components_t20,
        volume_concentration_t20=volume_concentration_t20,
    )


def _count_components(mask: np.ndarray) -> int:
    """Count connected components in a binary mask."""
    _, n = cc3d.connected_components(mask.astype(np.uint8), connectivity=26, return_N=True)
    return n


def _compute_volume_concentration(mask: np.ndarray) -> float:
    """Compute concentration: largest component volume / total volume."""
    labels, n = cc3d.connected_components(mask.astype(np.uint8), connectivity=26, return_N=True)
    if n == 0:
        return 0.0

    stats = cc3d.statistics(labels)
    voxel_counts = stats["voxel_counts"]

    # Index 0 is background
    if len(voxel_counts) <= 1:
        return 0.0

    total_vol = sum(voxel_counts[1:])
    max_vol = max(voxel_counts[1:])

    return max_vol / total_vol if total_vol > 0 else 0.0


# =============================================================================
# TARGETED TOPO ENHANCEMENT (2026-02-04)
# =============================================================================


def is_topo_at_risk(probs: np.ndarray) -> tuple[bool, TopoFeatures]:
    """Check if a sample is at risk for poor topology scores.

    Based on 30-sample validation (2026-02-04), samples with these characteristics
    tend to have poor topo scores and benefit from targeted threshold adjustment:
    - Very low decisive ratio (<0.35)
    - Low decisive ratio (<0.45) combined with high fragmentation (>100 components)

    Args:
        probs: 3D probability array

    Returns:
        Tuple of (is_at_risk, features)

    """
    features = extract_topo_features(probs)
    at_risk = features.decisive_ratio < 0.35 or (
        features.decisive_ratio < 0.45 and features.num_components_t20 > 100
    )
    return at_risk, features


def get_topo_optimized_threshold(features: TopoFeatures) -> float:
    """Get the optimal threshold for at-risk samples to maximize topo score.

    Rules derived from empirical testing on 5 at-risk samples (2026-02-04):
    - High kurtosis (>4.0): Use low threshold (0.10) to capture structure
    - Very high fragmentation (>150 components): Keep safe at 0.20
    - Low kurtosis (<2.0) + moderate components (<100): Use higher threshold (0.30)
    - Otherwise: Default to 0.20

    Validated on all 30 samples: +0.0377 total improvement, 0 regressions.

    Args:
        features: TopoFeatures named tuple from extract_topo_features()

    Returns:
        Optimal threshold value (0.10, 0.20, or 0.30)

    """
    if features.prob_kurtosis > 4.0:
        return 0.10  # e.g., sample 1577382633
    if features.num_components_t20 > 150:
        return 0.20  # Safe for highly fragmented (e.g., 3878283235, 648443467)
    if features.prob_kurtosis < 2.0 and features.num_components_t20 < 100:
        return 0.30  # Higher for flat distributions (e.g., 2203617984, 3602356228)
    return 0.20  # Default safe


def apply_targeted_topo_enhancement(
    probs: np.ndarray,
    default_threshold: float = 0.20,
    return_info: bool = False,
) -> np.ndarray | tuple[np.ndarray, dict]:
    """Apply targeted topo enhancement only for at-risk samples.

    This function identifies samples likely to have poor topology scores and
    applies a topo-optimized threshold. For samples that are not at-risk,
    it returns a simple threshold at the default value.

    Validated improvement on 30 samples:
    - 5 at-risk samples identified
    - 3 improved (total +0.0377 score improvement)
    - 0 regressed
    - 27 unchanged (already performing well)

    Args:
        probs: 3D probability array
        default_threshold: Threshold to use for non-risk samples (default 0.20)
        return_info: If True, return additional info dict

    Returns:
        Binary mask (np.uint8), or tuple of (mask, info_dict) if return_info=True

    """
    at_risk, features = is_topo_at_risk(probs)

    if at_risk:
        opt_threshold = get_topo_optimized_threshold(features)
        binary = (probs > opt_threshold).astype(np.uint8)
        strategy = f"topo_optimized_t={opt_threshold:.2f}"
    else:
        binary = (probs > default_threshold).astype(np.uint8)
        opt_threshold = default_threshold
        strategy = "default"

    if return_info:
        info = {
            "at_risk": at_risk,
            "decisive_ratio": features.decisive_ratio,
            "num_components_t20": features.num_components_t20,
            "prob_kurtosis": features.prob_kurtosis,
            "threshold": opt_threshold,
            "strategy": strategy,
        }
        return binary, info

    return binary


# =============================================================================
# STRATEGY RECOMMENDATION
# =============================================================================


def recommend_strategy(probs: np.ndarray) -> tuple[str, dict]:
    """Recommend a post-processing strategy based on sample characteristics.

    Updated 2026-02-03 with refined thresholds from 80-sample analysis.
    Updated again 2026-02-03: More conservative strategies after testing.

    Args:
        probs: 3D probability array

    Returns:
        Tuple of (strategy_name, params_dict)

    """
    features = extract_topo_features(probs)

    # Priority 1: VERY LOW DECISIVE RATIO (<0.35) - worst topo scores
    # These samples have mean topo=0.17 (worst category)
    # Key insight: Don't try to recover weak regions - focus on confident structure
    if features.decisive_ratio < 0.35:
        if features.num_components_t20 > 100:
            # Uncertain AND fragmented - most difficult case
            # Strategy: Keep ONLY the largest components, use high threshold
            return (
                "salvage_largest",
                {
                    "threshold": 0.25,  # High threshold to reduce noise
                    "min_size": 1500,  # Only substantial components
                    "closing_radius": 2,  # Moderate closing
                    "keep_k": 3,  # Keep only top-3 largest - better for topo
                },
            )
        # Uncertain but not too fragmented - conservative threshold
        # Don't use hysteresis - it extends into too much noise
        return (
            "conservative_threshold",
            {
                "threshold": 0.20,  # Standard threshold - proven to work
                "min_size": 800,
                "closing_radius": 1,
                "fill_holes": True,
            },
        )

    # Priority 2: HIGH FRAGMENTATION (>100 components at t=0.20)
    # 23 samples with mean topo=0.31 - still problematic
    if features.num_components_t20 > 100:
        if features.decisive_ratio > 0.55:
            # Fragmented but confident - use threshold that works
            return (
                "confident_cleanup",
                {
                    "threshold": 0.18,  # Slightly lower than 0.20
                    "min_size": 800,
                    "closing_radius": 1,
                    "keep_k": 15,
                },
            )
        # Moderately uncertain + fragmented
        return (
            "moderate_cleanup",
            {
                "threshold": 0.20,
                "min_size": 1000,
                "closing_radius": 2,
                "keep_k": 10,
            },
        )

    # Priority 3: HIGH CONFIDENCE (decisive_ratio > 0.60)
    # These samples already work well - use conservative thresholds
    if features.decisive_ratio > 0.60:
        if features.prob_kurtosis > 5.0:
            # Very peaked distribution - threshold at 0.15 works
            return (
                "low_threshold_clean",
                {
                    "threshold": 0.15,  # Raised from 0.10
                    "min_size": 500,
                    "fill_holes": True,
                },
            )
        if features.num_components_t20 < 60:
            # Good quality, low fragmentation - simple threshold
            return (
                "simple_clean",
                {
                    "threshold": 0.18,  # Conservative
                    "min_size": 500,
                    "fill_holes": True,
                },
            )
        # Good quality but some fragmentation
        return (
            "simple_clean",
            {
                "threshold": 0.18,
                "min_size": 600,
                "fill_holes": True,
            },
        )

    # Priority 4: MODERATE FRAGMENTATION (80-100 components)
    if features.num_components_t20 > 80:
        return (
            "component_filter",
            {
                "threshold": 0.18,
                "min_size": 700,
                "min_z_span": 25,
                "closing_radius": 1,
            },
        )

    # Priority 5: HIGH CONCENTRATION (one component dominates)
    if features.volume_concentration_t20 > 0.7:
        return (
            "balanced_threshold",
            {
                "threshold": 0.15,  # Lower to catch secondary
                "min_size": 500,
                "fill_holes": True,
            },
        )

    # Default: balanced approach
    return (
        "balanced",
        {
            "threshold": 0.18,
            "min_size": 500,
            "closing_radius": 1,
            "fill_holes": True,
        },
    )


# =============================================================================
# POST-PROCESSING STRATEGIES
# =============================================================================


def _apply_aggressive_cleanup(probs: np.ndarray, params: dict) -> np.ndarray:
    """Strategy for highly fragmented, uncertain samples."""
    threshold = params.get("threshold", 0.25)
    min_size = params.get("min_size", 2000)
    closing_radius = params.get("closing_radius", 3)
    keep_k = params.get("keep_k", 10)

    # 1. High threshold to reduce noise
    mask = (probs > threshold).astype(np.uint8)

    # 2. Morphological closing to merge nearby components
    if closing_radius > 0:
        struct = generate_binary_structure(3, 2)  # 26-connectivity
        mask = binary_closing(mask, structure=struct, iterations=closing_radius)
        mask = mask.astype(np.uint8)

    # 3. Keep only K largest components
    labels, n = cc3d.largest_k(mask, k=keep_k, connectivity=26, return_N=True)
    mask = (labels > 0).astype(np.uint8)

    # 4. Remove remaining small components
    labels = cc3d.connected_components(mask, connectivity=26)
    labels = cc3d.dust(labels, threshold=min_size, connectivity=26)
    mask = (labels > 0).astype(np.uint8)

    return mask


def _apply_moderate_cleanup(probs: np.ndarray, params: dict) -> np.ndarray:
    """Strategy for moderately fragmented samples."""
    threshold = params.get("threshold", 0.20)
    min_size = params.get("min_size", 1000)
    closing_radius = params.get("closing_radius", 2)
    keep_k = params.get("keep_k", 15)

    # 1. Threshold
    mask = (probs > threshold).astype(np.uint8)

    # 2. Light closing
    if closing_radius > 0:
        struct = generate_binary_structure(3, 1)  # 6-connectivity for gentler closing
        mask = binary_closing(mask, structure=struct, iterations=closing_radius)
        mask = mask.astype(np.uint8)

    # 3. Keep top-K components
    labels, _ = cc3d.largest_k(mask, k=keep_k, connectivity=26, return_N=True)
    mask = (labels > 0).astype(np.uint8)

    # 4. Size filter
    labels = cc3d.connected_components(mask, connectivity=26)
    labels = cc3d.dust(labels, threshold=min_size, connectivity=26)

    return (labels > 0).astype(np.uint8)


def _apply_salvage_largest(probs: np.ndarray, params: dict) -> np.ndarray:
    """Strategy for the most difficult samples: low decisive + high fragmentation.

    Instead of trying to recover weak connections, we focus on keeping only
    the most confident structures. This often produces better topo scores
    than aggressive recovery attempts.
    """
    threshold = params.get("threshold", 0.25)
    min_size = params.get("min_size", 1500)
    closing_radius = params.get("closing_radius", 2)
    keep_k = params.get("keep_k", 3)

    # 1. High threshold to get only confident regions
    mask = (probs > threshold).astype(np.uint8)

    # 2. Moderate closing to connect nearby confident regions
    if closing_radius > 0:
        struct = generate_binary_structure(3, 1)
        mask = binary_closing(mask, structure=struct, iterations=closing_radius)
        mask = mask.astype(np.uint8)

    # 3. Keep only top-K largest - for topo, fewer is often better
    labels, n = cc3d.largest_k(mask, k=keep_k, connectivity=26, return_N=True)
    mask = (labels > 0).astype(np.uint8)

    # 4. Final size filter
    labels = cc3d.connected_components(mask, connectivity=26)
    labels = cc3d.dust(labels, threshold=min_size, connectivity=26)

    return (labels > 0).astype(np.uint8)


def _apply_conservative_threshold(probs: np.ndarray, params: dict) -> np.ndarray:
    """Conservative strategy for uncertain samples without high fragmentation.

    Uses standard threshold (0.20) which is proven to work well on average.
    Avoids hysteresis which can extend into noisy regions.
    """
    threshold = params.get("threshold", 0.20)
    min_size = params.get("min_size", 800)
    closing_radius = params.get("closing_radius", 1)
    fill_holes = params.get("fill_holes", True)

    mask = (probs > threshold).astype(np.uint8)

    if closing_radius > 0:
        struct = generate_binary_structure(3, 1)
        mask = binary_closing(mask, structure=struct, iterations=closing_radius)
        mask = mask.astype(np.uint8)

    if fill_holes:
        mask = binary_fill_holes(mask).astype(np.uint8)

    labels = cc3d.connected_components(mask, connectivity=26)
    labels = cc3d.dust(labels, threshold=min_size, connectivity=26)

    return (labels > 0).astype(np.uint8)


def _apply_hysteresis_recovery(probs: np.ndarray, params: dict) -> np.ndarray:
    """Strategy for uncertain samples - recover weak connections."""
    high_thresh = params.get("high_thresh", 0.25)
    low_thresh = params.get("low_thresh", 0.08)
    min_size = params.get("min_size", 500)
    bridge_gaps = params.get("bridge_gaps", True)

    # 1. Get high-confidence seeds
    high_mask = probs > high_thresh

    # 2. Get extended low-threshold mask
    low_mask = probs > low_thresh

    # 3. Label low-threshold components
    labeled, num_features = scipy_label(low_mask)

    # 4. Find components containing high-confidence seeds
    high_labels = np.unique(labeled[high_mask])
    high_labels = high_labels[high_labels != 0]

    # 5. Keep only seeded components
    mask = np.isin(labeled, high_labels).astype(np.uint8)

    # 6. Optional: Bridge small gaps
    if bridge_gaps:
        struct = generate_binary_structure(3, 1)  # 6-connectivity
        dilated = binary_dilation(mask, structure=struct, iterations=1)
        # Only keep dilated regions that are in low-threshold mask
        mask = (dilated & low_mask).astype(np.uint8)

    # 7. Remove small components
    labels = cc3d.connected_components(mask, connectivity=26)
    labels = cc3d.dust(labels, threshold=min_size, connectivity=26)

    return (labels > 0).astype(np.uint8)


def _apply_light_hysteresis(probs: np.ndarray, params: dict) -> np.ndarray:
    """Strategy for good quality samples with peaked distributions."""
    high_thresh = params.get("high_thresh", 0.18)
    low_thresh = params.get("low_thresh", 0.12)
    min_size = params.get("min_size", 500)

    # Simple hysteresis
    high_mask = probs > high_thresh
    low_mask = probs > low_thresh

    labeled, _ = scipy_label(low_mask)
    high_labels = np.unique(labeled[high_mask])
    high_labels = high_labels[high_labels != 0]

    mask = np.isin(labeled, high_labels).astype(np.uint8)

    # Size filter
    labels = cc3d.connected_components(mask, connectivity=26)
    labels = cc3d.dust(labels, threshold=min_size, connectivity=26)

    return (labels > 0).astype(np.uint8)


def _apply_simple_clean(probs: np.ndarray, params: dict) -> np.ndarray:
    """Strategy for already-good samples - minimal processing."""
    threshold = params.get("threshold", 0.20)
    min_size = params.get("min_size", 500)
    fill_holes = params.get("fill_holes", True)

    mask = (probs > threshold).astype(np.uint8)

    if fill_holes:
        mask = binary_fill_holes(mask).astype(np.uint8)

    labels = cc3d.connected_components(mask, connectivity=26)
    labels = cc3d.dust(labels, threshold=min_size, connectivity=26)

    return (labels > 0).astype(np.uint8)


def _apply_component_filter(probs: np.ndarray, params: dict) -> np.ndarray:
    """Strategy for moderate fragmentation - filter by shape."""
    threshold = params.get("threshold", 0.20)
    min_size = params.get("min_size", 800)
    min_z_span = params.get("min_z_span", 30)
    closing_radius = params.get("closing_radius", 1)

    mask = (probs > threshold).astype(np.uint8)

    # Light closing
    if closing_radius > 0:
        struct = generate_binary_structure(3, 1)
        mask = binary_closing(mask, structure=struct, iterations=closing_radius)
        mask = mask.astype(np.uint8)

    # Label and filter by shape
    labels, n_labels = cc3d.connected_components(mask, connectivity=26, return_N=True)

    if n_labels == 0:
        return mask

    stats = cc3d.statistics(labels)
    keep_labels = []

    for label_id in range(1, n_labels + 1):
        if label_id >= len(stats["voxel_counts"]):
            continue

        voxel_count = stats["voxel_counts"][label_id]
        if voxel_count < min_size:
            continue

        # Get bounding box for z-span
        bbox = stats["bounding_boxes"][label_id]
        z_span = bbox[0].stop - bbox[0].start

        if z_span >= min_z_span:
            keep_labels.append(label_id)

    return np.isin(labels, keep_labels).astype(np.uint8)


def _apply_balanced_threshold(probs: np.ndarray, params: dict) -> np.ndarray:
    """Strategy for high-concentration samples."""
    threshold = params.get("threshold", 0.18)
    min_size = params.get("min_size", 500)
    fill_holes = params.get("fill_holes", True)

    mask = (probs > threshold).astype(np.uint8)

    if fill_holes:
        mask = binary_fill_holes(mask).astype(np.uint8)

    labels = cc3d.connected_components(mask, connectivity=26)
    labels = cc3d.dust(labels, threshold=min_size, connectivity=26)

    return (labels > 0).astype(np.uint8)


def _apply_balanced(probs: np.ndarray, params: dict) -> np.ndarray:
    """Default balanced strategy."""
    threshold = params.get("threshold", 0.20)
    min_size = params.get("min_size", 500)
    closing_radius = params.get("closing_radius", 1)
    fill_holes = params.get("fill_holes", True)

    mask = (probs > threshold).astype(np.uint8)

    if closing_radius > 0:
        struct = generate_binary_structure(3, 1)
        mask = binary_closing(mask, structure=struct, iterations=closing_radius)
        mask = mask.astype(np.uint8)

    if fill_holes:
        mask = binary_fill_holes(mask).astype(np.uint8)

    labels = cc3d.connected_components(mask, connectivity=26)
    labels = cc3d.dust(labels, threshold=min_size, connectivity=26)

    return (labels > 0).astype(np.uint8)


def _apply_ultra_aggressive_cleanup(probs: np.ndarray, params: dict) -> np.ndarray:
    """Strategy for the most difficult samples: uncertain AND fragmented.

    These samples (decisive_ratio < 0.35 AND >150 components) have the worst
    topo scores. We use very aggressive filtering to salvage what we can.
    """
    threshold = params.get("threshold", 0.30)
    min_size = params.get("min_size", 3000)
    closing_radius = params.get("closing_radius", 4)
    keep_k = params.get("keep_k", 5)

    # 1. Very high threshold
    mask = (probs > threshold).astype(np.uint8)

    # 2. Heavy morphological closing to merge nearby fragments
    if closing_radius > 0:
        struct = generate_binary_structure(3, 2)  # 26-connectivity
        mask = binary_closing(mask, structure=struct, iterations=closing_radius)
        mask = mask.astype(np.uint8)

    # 3. Keep only K largest components
    labels, n = cc3d.largest_k(mask, k=keep_k, connectivity=26, return_N=True)
    mask = (labels > 0).astype(np.uint8)

    # 4. Final size filter
    labels = cc3d.connected_components(mask, connectivity=26)
    labels = cc3d.dust(labels, threshold=min_size, connectivity=26)

    return (labels > 0).astype(np.uint8)


def _apply_confident_cleanup(probs: np.ndarray, params: dict) -> np.ndarray:
    """Strategy for fragmented but confident samples.

    When decisive_ratio > 0.55 but components > 100, we can use a lower
    threshold and keep more components since predictions are reliable.
    """
    threshold = params.get("threshold", 0.15)
    min_size = params.get("min_size", 800)
    closing_radius = params.get("closing_radius", 2)
    keep_k = params.get("keep_k", 20)

    # 1. Lower threshold since we're confident
    mask = (probs > threshold).astype(np.uint8)

    # 2. Moderate closing
    if closing_radius > 0:
        struct = generate_binary_structure(3, 1)  # 6-connectivity for gentler closing
        mask = binary_closing(mask, structure=struct, iterations=closing_radius)
        mask = mask.astype(np.uint8)

    # 3. Keep top-K components (more than usual since we trust predictions)
    labels, _ = cc3d.largest_k(mask, k=keep_k, connectivity=26, return_N=True)
    mask = (labels > 0).astype(np.uint8)

    # 4. Size filter
    labels = cc3d.connected_components(mask, connectivity=26)
    labels = cc3d.dust(labels, threshold=min_size, connectivity=26)

    return (labels > 0).astype(np.uint8)


def _apply_low_threshold_clean(probs: np.ndarray, params: dict) -> np.ndarray:
    """Strategy for high confidence + high kurtosis samples.

    These samples have very peaked distributions - the model is confident
    and predictions are concentrated. Use low threshold to capture structure.
    """
    threshold = params.get("threshold", 0.10)
    min_size = params.get("min_size", 400)
    fill_holes = params.get("fill_holes", True)

    mask = (probs > threshold).astype(np.uint8)

    if fill_holes:
        mask = binary_fill_holes(mask).astype(np.uint8)

    # Light cleanup
    labels = cc3d.connected_components(mask, connectivity=26)
    labels = cc3d.dust(labels, threshold=min_size, connectivity=26)

    return (labels > 0).astype(np.uint8)


# Strategy dispatcher
STRATEGY_FUNCTIONS = {
    "ultra_aggressive_cleanup": _apply_ultra_aggressive_cleanup,
    "aggressive_cleanup": _apply_aggressive_cleanup,
    "confident_cleanup": _apply_confident_cleanup,
    "moderate_cleanup": _apply_moderate_cleanup,
    "hysteresis_recovery": _apply_hysteresis_recovery,
    "light_hysteresis": _apply_light_hysteresis,
    "low_threshold_clean": _apply_low_threshold_clean,
    "simple_clean": _apply_simple_clean,
    "component_filter": _apply_component_filter,
    "balanced_threshold": _apply_balanced_threshold,
    "balanced": _apply_balanced,
    "salvage_largest": _apply_salvage_largest,
    "conservative_threshold": _apply_conservative_threshold,
}


# =============================================================================
# MAIN API
# =============================================================================


def adaptive_topo_postprocess(
    probs: np.ndarray,
    strategy: str = "auto",
    params: dict | None = None,
    return_info: bool = False,
) -> np.ndarray | tuple[np.ndarray, dict]:
    """Apply sample-specific post-processing optimized for topological score.

    This function analyzes the probability distribution characteristics of each
    sample and applies a targeted post-processing strategy designed to maximize
    the topological score (betti-matching F1).

    Args:
        probs: 3D probability array (D, H, W) or (2, D, H, W) with values 0-1
        strategy: Post-processing strategy to use:
            - "auto": Automatically select based on sample features (recommended)
            - "aggressive_cleanup": For highly fragmented, uncertain samples
            - "moderate_cleanup": For moderately fragmented samples
            - "hysteresis_recovery": For uncertain samples with weak connections
            - "light_hysteresis": For good samples with peaked distributions
            - "simple_clean": For already-good samples
            - "component_filter": For moderate fragmentation with shape filtering
            - "balanced_threshold": For high-concentration samples
            - "balanced": Default balanced approach
        params: Optional dict of parameters to override strategy defaults
        return_info: If True, also return dict with features and strategy used

    Returns:
        Binary mask (uint8). If return_info=True, also returns info dict.

    Example:
        >>> probs = np.load("prediction.npz")["probabilities"]
        >>> mask = adaptive_topo_postprocess(probs)

        >>> # With info
        >>> mask, info = adaptive_topo_postprocess(probs, return_info=True)
        >>> print(f"Strategy used: {info['strategy']}")

    """
    # Handle multi-channel input
    if probs.ndim == 4 and probs.shape[0] == 2:
        probs = probs[1]  # Foreground channel

    # Get features
    features = extract_topo_features(probs)

    # Determine strategy
    if strategy == "auto":
        strategy, auto_params = recommend_strategy(probs)
        # Merge user params with auto params (user overrides)
        if params:
            auto_params.update(params)
        params = auto_params
    elif params is None:
        params = {}

    # Apply strategy
    if strategy not in STRATEGY_FUNCTIONS:
        raise ValueError(
            f"Unknown strategy: {strategy}. Available: {list(STRATEGY_FUNCTIONS.keys())}",
        )

    mask = STRATEGY_FUNCTIONS[strategy](probs, params)

    if return_info:
        info = {
            "strategy": strategy,
            "params": params,
            "features": features._asdict(),
        }
        return mask, info

    return mask


def get_strategy_for_features(features: TopoFeatures) -> tuple[str, dict]:
    """Get recommended strategy from pre-computed features.

    Useful when you've already computed features and want to know the strategy.
    Updated 2026-02-03 to match recommend_strategy logic with conservative strategies.

    Args:
        features: TopoFeatures named tuple

    Returns:
        Tuple of (strategy_name, params_dict)

    """
    # Priority 1: VERY LOW DECISIVE RATIO (<0.35) - worst topo scores
    # These samples have mean topo=0.17 (worst category)
    # Key insight: Don't try to recover weak regions - focus on confident structure
    if features.decisive_ratio < 0.35:
        if features.num_components_t20 > 100:
            # Uncertain AND fragmented - most difficult case
            # Strategy: Keep ONLY the largest components, use high threshold
            return (
                "salvage_largest",
                {
                    "threshold": 0.25,  # High threshold to reduce noise
                    "min_size": 1500,  # Only substantial components
                    "closing_radius": 2,  # Moderate closing
                    "keep_k": 3,  # Keep only top-3 largest - better for topo
                },
            )
        # Uncertain but not too fragmented - conservative threshold
        # Don't use hysteresis - it extends into too much noise
        return (
            "conservative_threshold",
            {
                "threshold": 0.20,  # Standard threshold - proven to work
                "min_size": 800,
                "closing_radius": 1,
                "fill_holes": True,
            },
        )

    # Priority 2: HIGH FRAGMENTATION (>100 components at t=0.20)
    if features.num_components_t20 > 100:
        if features.decisive_ratio > 0.55:
            # Fragmented but confident - use threshold that works
            return (
                "confident_cleanup",
                {
                    "threshold": 0.18,  # Slightly lower than 0.20
                    "min_size": 800,
                    "closing_radius": 1,
                    "keep_k": 15,
                },
            )
        # Moderately uncertain + fragmented
        return (
            "moderate_cleanup",
            {
                "threshold": 0.20,
                "min_size": 1000,
                "closing_radius": 2,
                "keep_k": 10,
            },
        )

    # Priority 3: HIGH CONFIDENCE (decisive_ratio > 0.60)
    # These samples already work well - use conservative thresholds
    if features.decisive_ratio > 0.60:
        if features.prob_kurtosis > 5.0:
            # Very peaked distribution - threshold at 0.15 works
            return (
                "low_threshold_clean",
                {
                    "threshold": 0.15,  # Raised from 0.10
                    "min_size": 500,
                    "fill_holes": True,
                },
            )
        if features.num_components_t20 < 60:
            # Good quality, low fragmentation - simple threshold
            return (
                "simple_clean",
                {
                    "threshold": 0.18,  # Conservative
                    "min_size": 500,
                    "fill_holes": True,
                },
            )
        # Good quality but some fragmentation
        return (
            "simple_clean",
            {
                "threshold": 0.18,
                "min_size": 600,
                "fill_holes": True,
            },
        )

    # Priority 4: MODERATE FRAGMENTATION (80-100 components)
    if features.num_components_t20 > 80:
        return (
            "component_filter",
            {
                "threshold": 0.18,
                "min_size": 700,
                "min_z_span": 25,
                "closing_radius": 1,
            },
        )

    # Priority 5: HIGH CONCENTRATION
    if features.volume_concentration_t20 > 0.7:
        return (
            "balanced_threshold",
            {
                "threshold": 0.15,
                "min_size": 500,
                "fill_holes": True,
            },
        )

    return (
        "balanced",
        {
            "threshold": 0.18,
            "min_size": 500,
            "closing_radius": 1,
            "fill_holes": True,
        },
    )
