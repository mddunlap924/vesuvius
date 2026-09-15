"""Analyze prediction probability maps to extract features for adaptive thresholding.

This script analyzes NPZ prediction files to extract metrics that may correlate with
the optimal threshold choice per sample. The goal is to identify which samples benefit
from lower vs higher thresholds based on probability distribution characteristics.

Key hypotheses for threshold selection:
1. Samples with higher mean/median foreground probability may need higher thresholds
2. Samples with more bimodal distributions may be easier to threshold
3. Samples with more gradual probability transitions may need different handling
4. Spatial features (compactness, continuity) may inform threshold choice

Features extracted:
- Probability distribution statistics (mean, std, percentiles, skewness, kurtosis)
- Bimodality metrics (Hartigan's dip test proxy, peak analysis)
- Spatial coherence metrics (gradient magnitudes, local variance)
- Thresholded volume characteristics at various thresholds
- Z-continuity of high-probability regions

USAGE FOR ADAPTIVE THRESHOLDING:
================================
To use the adaptive threshold prediction in post-processing:

    from analyze_prediction_probabilities import adaptive_threshold_postprocess

    # Apply adaptive threshold to a probability volume
    binary_mask = adaptive_threshold_postprocess(probs, method="learned")

Or for use with file paths:

    from analyze_prediction_probabilities import predict_threshold_for_file

    threshold, features = predict_threshold_for_file(pred_path, method="learned")
"""

import json
import os
import warnings
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import cpu_count
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage
from scipy.ndimage import (
    binary_dilation,
    binary_erosion,
    distance_transform_edt,
    gaussian_filter,
    uniform_filter,
)
from scipy.ndimage import label as scipy_label
from scipy.stats import entropy, kurtosis, skew
from tqdm import tqdm

# Suppress warnings
warnings.filterwarnings("ignore")


# =============================================================================
# ADAPTIVE THRESHOLDING - MAIN API
# =============================================================================


def adaptive_threshold_postprocess(
    probs: np.ndarray,
    method: str = "learned",
    min_size: int = 500,
) -> np.ndarray:
    """Apply adaptive thresholding to a probability volume.

    This is the main entry point for using adaptive thresholding in post-processing.
    It analyzes the probability distribution characteristics and selects an
    appropriate threshold for this specific sample.

    Args:
        probs: 3D array of foreground probabilities (values 0-1)
               Can be shape (D, H, W) or (2, D, H, W) where channel 1 is foreground
        method: Threshold prediction method:
            - "learned": Uses learned decision boundaries (recommended)
            - "regression": Linear regression on features
            - "rule_based": Simple if/else rules
            - "hybrid": Combination of regression and rules
        min_size: Minimum component size (smaller components removed)

    Returns:
        Binary mask (uint8) after adaptive thresholding

    Example:
        >>> probs = np.load("prediction.npz")["probabilities"]
        >>> binary_mask = adaptive_threshold_postprocess(probs)

    """
    # Handle multi-channel input
    if probs.ndim == 4 and probs.shape[0] == 2:
        probs = probs[1]  # Foreground channel

    # Predict optimal threshold
    threshold = predict_optimal_threshold(probs, method=method)

    # Apply threshold
    binary = (probs > threshold).astype(np.uint8)

    # Remove small components
    if min_size > 0:
        labeled, num_components = scipy_label(binary)
        if num_components > 0:
            comp_sizes = ndimage.sum(binary, labeled, range(1, num_components + 1))
            for i, size in enumerate(comp_sizes, 1):
                if size < min_size:
                    binary[labeled == i] = 0

    return binary


def get_adaptive_threshold(probs: np.ndarray, method: str = "learned") -> float:
    """Get the adaptive threshold value for a probability volume (without applying it).

    This is useful when you need just the threshold value for logging or
    combining with other post-processing steps.

    Args:
        probs: 3D array of foreground probabilities
        method: Threshold prediction method

    Returns:
        Predicted optimal threshold value (float between 0.06 and 0.25)

    """
    if probs.ndim == 4 and probs.shape[0] == 2:
        probs = probs[1]
    return predict_optimal_threshold(probs, method=method)


def adaptive_hysteresis_postprocess(
    probs: np.ndarray,
    min_size: int = 500,
    use_bridge_recovery: bool = True,
) -> np.ndarray:
    """Apply adaptive hysteresis thresholding to improve topology.

    This method is designed to improve the topological score (topo_dim0) by
    recovering weak connections that would be broken by simple thresholding.

    Strategy based on 80-sample analysis (2026-02-02):
    - For high-kurtosis samples (peaked distributions), the model has clear
      high-confidence regions but connections between them may be weak
    - Hysteresis recovers these weak bridges by using a LOW threshold to
      extend seed regions found at the HIGH threshold
    - Only voxels connected to high-confidence seeds are kept

    Key insight from analysis:
    - 17 samples showed topo gain >2% at 0.10 threshold with <1.5% final loss
    - These samples had: high kurtosis (4.98), low frac_medium_conf (0.034)
    - Hysteresis can capture this benefit without the full final score loss

    Args:
        probs: 3D array of foreground probabilities (values 0-1)
               Can be shape (D, H, W) or (2, D, H, W) where channel 1 is foreground
        min_size: Minimum component size (smaller components removed)
        use_bridge_recovery: If True, applies additional morphological dilation
                           to bridge small gaps for high-kurtosis samples

    Returns:
        Binary mask (uint8) after adaptive hysteresis thresholding

    Example:
        >>> probs = np.load("prediction.npz")["probabilities"]
        >>> binary_mask = adaptive_hysteresis_postprocess(probs)

    """
    # Handle multi-channel input
    if probs.ndim == 4 and probs.shape[0] == 2:
        probs = probs[1]  # Foreground channel

    flat_probs = probs.flatten()

    # Compute key features
    prob_mean = np.mean(flat_probs)
    prob_kurtosis_val = kurtosis(flat_probs)

    # Decide on hysteresis parameters based on probability features
    # High kurtosis = peaked distribution = clear foreground but potentially broken connections
    if prob_kurtosis_val > 4.5 and prob_mean < 0.10:
        # Very peaked distribution - use aggressive hysteresis
        # These samples showed biggest topo gains in analysis
        high_thresh = 0.12  # Strict seeds
        low_thresh = 0.05  # Extend to weak connections
        apply_bridge = use_bridge_recovery
    elif prob_kurtosis_val > 4.0 and prob_mean < 0.12:
        # Moderately peaked - moderate hysteresis
        high_thresh = 0.15
        low_thresh = 0.07
        apply_bridge = use_bridge_recovery
    elif prob_kurtosis_val > 3.5:
        # Mild peaked distribution
        high_thresh = 0.18
        low_thresh = 0.08
        apply_bridge = False
    else:
        # Flat distribution - use standard thresholding with mild hysteresis
        high_thresh = 0.20
        low_thresh = 0.10
        apply_bridge = False

    # Apply hysteresis thresholding
    high_mask = probs > high_thresh
    low_mask = probs > low_thresh

    # Label connected components in the low threshold mask
    labeled, num_features = scipy_label(low_mask)

    # Find which components contain high-confidence pixels
    high_labels = np.unique(labeled[high_mask])
    high_labels = high_labels[high_labels != 0]  # Remove background

    # Keep only components that have high-confidence pixels
    binary = np.isin(labeled, high_labels).astype(np.uint8)

    # Optional: Apply morphological bridging for high-kurtosis samples
    # This helps connect components that are separated by very small gaps
    if apply_bridge and prob_kurtosis_val > 4.0:
        # Light dilation to bridge small gaps
        struct = ndimage.generate_binary_structure(3, 1)  # 6-connectivity
        dilated = binary_dilation(binary, structure=struct, iterations=1)

        # Re-threshold to clean up the dilation
        # Only keep dilated regions that overlap with low-threshold mask
        binary = (dilated & low_mask).astype(np.uint8)

    # Remove small components
    if min_size > 0:
        labeled, num_components = scipy_label(binary)
        if num_components > 0:
            comp_sizes = ndimage.sum(binary, labeled, range(1, num_components + 1))
            for i, size in enumerate(comp_sizes, 1):
                if size < min_size:
                    binary[labeled == i] = 0

    return binary


def predict_topo_optimal_threshold(probs: np.ndarray) -> tuple[float, float]:
    """Predict optimal LOW and HIGH thresholds for topo-focused hysteresis.

    This function is specifically designed to maximize topological score.
    Based on 80-sample analysis, samples with high kurtosis benefit most
    from lower thresholds for topo, while final score prefers 0.20.

    Args:
        probs: 3D array of foreground probabilities

    Returns:
        Tuple of (low_threshold, high_threshold) for hysteresis

    """
    flat_probs = probs.flatten()

    prob_mean = np.mean(flat_probs)
    prob_kurtosis_val = kurtosis(flat_probs)
    frac_medium_conf = ((flat_probs > 0.3) & (flat_probs <= 0.7)).mean()

    # Analysis showed:
    # - topo_thresh_0.10 is best for 36 samples (45% of dataset)
    # - These samples have: kurtosis=4.77, frac_medium_conf=0.040

    if prob_kurtosis_val > 5.0 and frac_medium_conf < 0.04:
        # Very peaked, very little uncertainty -> aggressive hysteresis
        return (0.05, 0.12)
    if prob_kurtosis_val > 4.2 and prob_mean < 0.12:
        # Peaked distribution with sparse predictions
        return (0.06, 0.15)
    if prob_kurtosis_val > 3.8:
        # Moderately peaked
        return (0.08, 0.18)
    # Standard distribution -> mild hysteresis
    return (0.10, 0.20)


# =============================================================================
# PROBABILITY DISTRIBUTION ANALYSIS
# =============================================================================


def analyze_probability_distribution(probs: np.ndarray) -> dict:
    """Analyze the probability distribution of the foreground predictions.

    Args:
        probs: 3D array of foreground probabilities (values 0-1)

    Returns:
        Dictionary of distribution statistics

    """
    # Flatten for distribution analysis
    flat_probs = probs.flatten()

    # Basic statistics
    stats = {
        "prob_mean": float(np.mean(flat_probs)),
        "prob_std": float(np.std(flat_probs)),
        "prob_median": float(np.median(flat_probs)),
        "prob_min": float(np.min(flat_probs)),
        "prob_max": float(np.max(flat_probs)),
    }

    # Percentiles
    percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    pcts = np.percentile(flat_probs, percentiles)
    for p, val in zip(percentiles, pcts):
        stats[f"prob_p{p}"] = float(val)

    # Interquartile range
    stats["prob_iqr"] = float(stats["prob_p75"] - stats["prob_p25"])

    # Skewness and kurtosis
    stats["prob_skewness"] = float(skew(flat_probs))
    stats["prob_kurtosis"] = float(kurtosis(flat_probs))

    # Coefficient of variation
    if stats["prob_mean"] > 1e-6:
        stats["prob_cv"] = float(stats["prob_std"] / stats["prob_mean"])
    else:
        stats["prob_cv"] = 0.0

    return stats


def analyze_bimodality(probs: np.ndarray, n_bins: int = 100) -> dict:
    """Analyze bimodality of the probability distribution.

    Bimodal distributions (peaks at low and high probability) are easier to threshold.
    Unimodal or multimodal distributions may require different strategies.

    Args:
        probs: 3D array of foreground probabilities
        n_bins: Number of histogram bins

    Returns:
        Dictionary of bimodality metrics

    """
    flat_probs = probs.flatten()

    # Compute histogram
    hist, bin_edges = np.histogram(flat_probs, bins=n_bins, range=(0, 1))
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Normalize histogram
    hist_norm = hist / hist.sum()

    # Find peaks (local maxima)
    peaks = []
    for i in range(1, len(hist) - 1):
        if hist[i] > hist[i - 1] and hist[i] > hist[i + 1] and hist[i] > hist.mean():
            peaks.append((i, hist[i], bin_centers[i]))

    # Bimodality coefficient (Sarle's)
    # BC = (skewness^2 + 1) / kurtosis
    # BC > 5/9 ≈ 0.555 suggests bimodality
    sk = skew(flat_probs)
    kurt = kurtosis(flat_probs)
    if kurt + 3 > 0:  # excess kurtosis, need to add 3 for raw kurtosis
        bimodality_coef = (sk**2 + 1) / (kurt + 3)
    else:
        bimodality_coef = 0.0

    # Valley detection between peaks
    # Find the minimum between the main peaks
    valley_depth = 0.0
    valley_position = 0.5
    if len(peaks) >= 2:
        # Sort peaks by height
        peaks_sorted = sorted(peaks, key=lambda x: x[1], reverse=True)
        # Get two highest peaks
        p1_idx = peaks_sorted[0][0]
        p2_idx = peaks_sorted[1][0]

        # Find valley between them
        start_idx = min(p1_idx, p2_idx)
        end_idx = max(p1_idx, p2_idx)

        if end_idx > start_idx:
            valley_idx = start_idx + np.argmin(hist[start_idx:end_idx])
            valley_depth = 1.0 - (hist[valley_idx] / max(hist[p1_idx], hist[p2_idx]))
            valley_position = bin_centers[valley_idx]

    # Entropy as measure of distribution spread
    hist_entropy = entropy(hist_norm + 1e-10)

    # Fraction of voxels near 0 and near 1
    fraction_low = (flat_probs < 0.1).mean()
    fraction_high = (flat_probs > 0.9).mean()
    fraction_middle = ((flat_probs >= 0.1) & (flat_probs <= 0.9)).mean()

    return {
        "num_peaks": len(peaks),
        "bimodality_coef": float(bimodality_coef),
        "valley_depth": float(valley_depth),
        "valley_position": float(valley_position),
        "hist_entropy": float(hist_entropy),
        "fraction_low_prob": float(fraction_low),
        "fraction_high_prob": float(fraction_high),
        "fraction_middle_prob": float(fraction_middle),
        "low_to_high_ratio": float(fraction_low / max(fraction_high, 1e-6)),
    }


def analyze_thresholded_volumes(probs: np.ndarray) -> dict:
    """Analyze characteristics at various threshold levels.

    Args:
        probs: 3D array of foreground probabilities

    Returns:
        Dictionary of threshold-based metrics

    """
    thresholds = [0.05, 0.08, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]
    total_voxels = probs.size

    stats = {}
    prev_count = total_voxels

    for thresh in thresholds:
        mask = probs > thresh
        count = mask.sum()
        volume_fraction = count / total_voxels

        # Volume change rate between thresholds
        if prev_count > 0:
            retention_rate = count / prev_count
        else:
            retention_rate = 0.0

        # Connected components at this threshold
        labeled, num_components = scipy_label(mask)

        # Component sizes
        if num_components > 0:
            comp_sizes = ndimage.sum(mask, labeled, range(1, num_components + 1))
            max_comp_size = comp_sizes.max() if len(comp_sizes) > 0 else 0
            mean_comp_size = comp_sizes.mean() if len(comp_sizes) > 0 else 0
        else:
            max_comp_size = 0
            mean_comp_size = 0

        stats[f"thresh_{thresh:.2f}_count"] = int(count)
        stats[f"thresh_{thresh:.2f}_volume_frac"] = float(volume_fraction)
        stats[f"thresh_{thresh:.2f}_num_components"] = int(num_components)
        stats[f"thresh_{thresh:.2f}_max_comp_size"] = int(max_comp_size)
        stats[f"thresh_{thresh:.2f}_mean_comp_size"] = float(mean_comp_size)
        stats[f"thresh_{thresh:.2f}_retention_rate"] = float(retention_rate)

        prev_count = count

    # Compute sensitivity: how much volume changes per threshold increment
    # Higher sensitivity = more uncertain predictions
    volume_fracs = [stats[f"thresh_{t:.2f}_volume_frac"] for t in thresholds]
    if len(volume_fracs) > 1:
        sensitivity = np.std(volume_fracs) / (np.mean(volume_fracs) + 1e-6)
    else:
        sensitivity = 0.0

    stats["threshold_sensitivity"] = float(sensitivity)

    # Find "elbow" threshold where volume drops most rapidly
    diffs = np.diff(volume_fracs)
    max_drop_idx = np.argmax(np.abs(diffs))
    stats["max_drop_threshold"] = float(thresholds[max_drop_idx])
    stats["max_drop_magnitude"] = float(abs(diffs[max_drop_idx]))

    return stats


# =============================================================================
# SPATIAL COHERENCE ANALYSIS
# =============================================================================


def analyze_spatial_coherence(probs: np.ndarray, downsample: int = 2) -> dict:
    """Analyze spatial coherence of probability predictions.

    Spatially coherent predictions (smooth probability fields) may be easier
    to threshold accurately than noisy predictions.

    Args:
        probs: 3D array of foreground probabilities
        downsample: Factor to downsample for speed

    Returns:
        Dictionary of spatial coherence metrics

    """
    # Downsample for speed
    if downsample > 1:
        probs = probs[::downsample, ::downsample, ::downsample]

    # Gradient magnitude (how rapidly probabilities change)
    gradz = np.diff(probs, axis=0)
    grady = np.diff(probs, axis=1)
    gradx = np.diff(probs, axis=2)

    # Trim to same shape for gradient magnitude
    min_shape = (
        min(gradz.shape[0], grady.shape[0], gradx.shape[0]),
        min(gradz.shape[1], grady.shape[1], gradx.shape[1]),
        min(gradz.shape[2], grady.shape[2], gradx.shape[2]),
    )
    gradz = gradz[: min_shape[0], : min_shape[1], : min_shape[2]]
    grady = grady[: min_shape[0], : min_shape[1], : min_shape[2]]
    gradx = gradx[: min_shape[0], : min_shape[1], : min_shape[2]]

    grad_mag = np.sqrt(gradz**2 + grady**2 + gradx**2)

    # Local variance using uniform filter
    local_mean = uniform_filter(probs.astype(np.float64), size=5)
    local_var = uniform_filter((probs.astype(np.float64) - local_mean) ** 2, size=5)

    # Focus on regions with non-negligible probability
    high_prob_mask = probs > 0.1

    stats = {
        "mean_gradient_magnitude": float(np.mean(grad_mag)),
        "std_gradient_magnitude": float(np.std(grad_mag)),
        "max_gradient_magnitude": float(np.max(grad_mag)),
        "mean_local_variance": float(np.mean(local_var)),
        "max_local_variance": float(np.max(local_var)),
    }

    if high_prob_mask.sum() > 0:
        # Gradient in high-probability regions
        high_prob_grad = grad_mag[
            : high_prob_mask.shape[0],
            : high_prob_mask.shape[1],
            : high_prob_mask.shape[2],
        ]
        hp_mask_trimmed = high_prob_mask[
            : high_prob_grad.shape[0],
            : high_prob_grad.shape[1],
            : high_prob_grad.shape[2],
        ]

        if hp_mask_trimmed.sum() > 0:
            stats["mean_gradient_high_prob"] = float(np.mean(high_prob_grad[hp_mask_trimmed]))
        else:
            stats["mean_gradient_high_prob"] = 0.0

        stats["mean_local_var_high_prob"] = float(np.mean(local_var[high_prob_mask]))
    else:
        stats["mean_gradient_high_prob"] = 0.0
        stats["mean_local_var_high_prob"] = 0.0

    # Edge sharpness: ratio of high gradients to mean gradient
    if stats["mean_gradient_magnitude"] > 0:
        high_grad_thresh = np.percentile(grad_mag, 95)
        edge_sharpness = high_grad_thresh / stats["mean_gradient_magnitude"]
    else:
        edge_sharpness = 0.0

    stats["edge_sharpness"] = float(edge_sharpness)

    return stats


def analyze_z_continuity(probs: np.ndarray, threshold: float = 0.15) -> dict:
    """Analyze z-axis continuity of predictions.

    Ribbons are expected to be continuous in z-direction. Predictions that
    show good z-continuity may be more reliable.

    Args:
        probs: 3D array of foreground probabilities
        threshold: Threshold for binary mask

    Returns:
        Dictionary of z-continuity metrics

    """
    mask = probs > threshold
    z_depth = probs.shape[0]

    # Calculate overlap between consecutive slices
    overlaps = []
    slice_areas = []

    for z in range(z_depth - 1):
        curr_slice = mask[z]
        next_slice = mask[z + 1]

        curr_area = curr_slice.sum()
        next_area = next_slice.sum()

        slice_areas.append(curr_area)

        if curr_area > 0 and next_area > 0:
            intersection = (curr_slice & next_slice).sum()
            union = (curr_slice | next_slice).sum()
            iou = intersection / union if union > 0 else 0
            overlaps.append(iou)
        elif curr_area == 0 and next_area == 0:
            overlaps.append(1.0)  # Both empty is consistent
        else:
            overlaps.append(0.0)  # One empty, one not = discontinuity

    slice_areas.append(mask[-1].sum())

    if overlaps:
        mean_overlap = np.mean(overlaps)
        min_overlap = np.min(overlaps)
        std_overlap = np.std(overlaps)
    else:
        mean_overlap = 0.0
        min_overlap = 0.0
        std_overlap = 0.0

    # Z-span: range of z where predictions exist
    z_presence = mask.any(axis=(1, 2))
    z_indices = np.where(z_presence)[0]

    if len(z_indices) > 0:
        z_span = z_indices[-1] - z_indices[0] + 1
        z_coverage = len(z_indices) / z_span if z_span > 0 else 0
        z_start = z_indices[0]
        z_end = z_indices[-1]
    else:
        z_span = 0
        z_coverage = 0
        z_start = -1
        z_end = -1

    # Slice area variance (ribbons should have consistent cross-section)
    slice_areas = np.array(slice_areas)
    nonzero_areas = slice_areas[slice_areas > 0]

    if len(nonzero_areas) > 1:
        area_cv = np.std(nonzero_areas) / np.mean(nonzero_areas)
    else:
        area_cv = 0.0

    return {
        "z_mean_overlap": float(mean_overlap),
        "z_min_overlap": float(min_overlap),
        "z_std_overlap": float(std_overlap),
        "z_span": int(z_span),
        "z_coverage": float(z_coverage),
        "z_start": int(z_start),
        "z_end": int(z_end),
        "slice_area_cv": float(area_cv),
    }


def analyze_border_connectivity(probs: np.ndarray, threshold: float = 0.15) -> dict:
    """Analyze border connectivity of predictions.

    Domain knowledge: Ribbons start/end at xy borders, not in the middle.
    Predictions respecting this may be more reliable.

    Args:
        probs: 3D array of foreground probabilities
        threshold: Threshold for binary mask

    Returns:
        Dictionary of border connectivity metrics

    """
    mask = probs > threshold

    if mask.sum() == 0:
        return {
            "touches_xy_border": False,
            "border_contact_ratio": 0.0,
            "touches_top": False,
            "touches_bottom": False,
            "touches_left": False,
            "touches_right": False,
        }

    # XY projection
    xy_proj = mask.any(axis=0)

    border_margin = 5
    h, w = xy_proj.shape

    touches_top = xy_proj[:border_margin, :].any()
    touches_bottom = xy_proj[-border_margin:, :].any()
    touches_left = xy_proj[:, :border_margin].any()
    touches_right = xy_proj[:, -border_margin:].any()

    border_contacts = sum([touches_top, touches_bottom, touches_left, touches_right])

    # Calculate how much of the prediction touches borders
    border_mask = np.zeros_like(xy_proj)
    border_mask[:border_margin, :] = True
    border_mask[-border_margin:, :] = True
    border_mask[:, :border_margin] = True
    border_mask[:, -border_margin:] = True

    border_contact_ratio = (xy_proj & border_mask).sum() / max(xy_proj.sum(), 1)

    return {
        "touches_xy_border": border_contacts > 0,
        "border_contact_count": int(border_contacts),
        "border_contact_ratio": float(border_contact_ratio),
        "touches_top": bool(touches_top),
        "touches_bottom": bool(touches_bottom),
        "touches_left": bool(touches_left),
        "touches_right": bool(touches_right),
    }


def analyze_confidence_regions(probs: np.ndarray) -> dict:
    """Analyze high-confidence vs low-confidence regions.

    Args:
        probs: 3D array of foreground probabilities

    Returns:
        Dictionary of confidence region metrics

    """
    # Define confidence levels
    very_high_conf = probs > 0.9
    high_conf = probs > 0.7
    medium_conf = (probs > 0.3) & (probs <= 0.7)
    low_conf = (probs > 0.1) & (probs <= 0.3)
    very_low_conf = probs <= 0.1

    total = probs.size

    stats = {
        "frac_very_high_conf": float(very_high_conf.sum() / total),
        "frac_high_conf": float(high_conf.sum() / total),
        "frac_medium_conf": float(medium_conf.sum() / total),
        "frac_low_conf": float(low_conf.sum() / total),
        "frac_very_low_conf": float(very_low_conf.sum() / total),
    }

    # Ratio of confident to uncertain predictions
    confident = (probs > 0.7) | (probs < 0.3)
    uncertain = (probs >= 0.3) & (probs <= 0.7)

    stats["confident_to_uncertain_ratio"] = float(
        confident.sum() / max(uncertain.sum(), 1),
    )

    # "Decisive" ratio: how much of the positive prediction is very confident
    positive_mask = probs > 0.15
    if positive_mask.sum() > 0:
        decisive_ratio = (probs[positive_mask] > 0.7).sum() / positive_mask.sum()
    else:
        decisive_ratio = 0.0

    stats["decisive_ratio"] = float(decisive_ratio)

    return stats


# =============================================================================
# COMPONENT ANALYSIS
# =============================================================================


def analyze_components_at_threshold(probs: np.ndarray, threshold: float = 0.15) -> dict:
    """Detailed analysis of connected components at a specific threshold.

    Args:
        probs: 3D array of foreground probabilities
        threshold: Threshold for binary mask

    Returns:
        Dictionary of component metrics

    """
    mask = probs > threshold
    labeled, num_components = scipy_label(mask)

    if num_components == 0:
        return {
            "num_components": 0,
            "total_volume": 0,
            "mean_comp_volume": 0,
            "max_comp_volume": 0,
            "volume_concentration": 0,
        }

    # Component sizes
    comp_sizes = ndimage.sum(mask, labeled, range(1, num_components + 1))
    comp_sizes = np.array(comp_sizes)

    total_volume = comp_sizes.sum()
    mean_volume = comp_sizes.mean()
    max_volume = comp_sizes.max()

    # Concentration: what fraction is in the largest component
    volume_concentration = max_volume / total_volume if total_volume > 0 else 0

    # Size distribution metrics
    if len(comp_sizes) > 1:
        size_std = comp_sizes.std()
        size_cv = size_std / mean_volume if mean_volume > 0 else 0
    else:
        size_std = 0
        size_cv = 0

    # Count small vs large components
    small_threshold = 1000  # voxels
    num_small = (comp_sizes < small_threshold).sum()
    num_large = (comp_sizes >= small_threshold).sum()

    return {
        "num_components": int(num_components),
        "total_volume": int(total_volume),
        "mean_comp_volume": float(mean_volume),
        "max_comp_volume": int(max_volume),
        "std_comp_volume": float(size_std),
        "cv_comp_volume": float(size_cv),
        "volume_concentration": float(volume_concentration),
        "num_small_components": int(num_small),
        "num_large_components": int(num_large),
    }


# =============================================================================
# OPTIMAL THRESHOLD ESTIMATION
# =============================================================================


def estimate_optimal_threshold(probs: np.ndarray) -> dict:
    """Estimate optimal threshold using various heuristics.

    Args:
        probs: 3D array of foreground probabilities

    Returns:
        Dictionary with threshold estimates from different methods

    """
    flat_probs = probs[probs > 0.01]  # Exclude very low values

    estimates = {}

    # Method 1: Otsu-like threshold on probabilities
    # Find threshold that maximizes inter-class variance
    best_thresh = 0.15
    best_variance = 0

    for thresh in np.arange(0.05, 0.50, 0.01):
        below = flat_probs[flat_probs <= thresh]
        above = flat_probs[flat_probs > thresh]

        if len(below) > 0 and len(above) > 0:
            w0 = len(below) / len(flat_probs)
            w1 = len(above) / len(flat_probs)
            var_between = w0 * w1 * (below.mean() - above.mean()) ** 2

            if var_between > best_variance:
                best_variance = var_between
                best_thresh = thresh

    estimates["otsu_threshold"] = float(best_thresh)

    # Method 2: Based on percentile of non-zero probabilities
    if len(flat_probs) > 0:
        estimates["p90_threshold"] = float(np.percentile(flat_probs, 90))
        estimates["p95_threshold"] = float(np.percentile(flat_probs, 95))
    else:
        estimates["p90_threshold"] = 0.15
        estimates["p95_threshold"] = 0.15

    # Method 3: Based on expected ribbon volume fraction
    # Typical ribbons occupy ~5-15% of the volume
    target_fractions = [0.05, 0.10, 0.15]
    for target_frac in target_fractions:
        # Find threshold that gives approximately this volume fraction
        target_count = int(probs.size * target_frac)
        sorted_probs = np.sort(probs.flatten())[::-1]  # Descending
        if len(sorted_probs) > target_count:
            estimates[f"target_{int(target_frac * 100)}pct_threshold"] = float(
                sorted_probs[target_count],
            )
        else:
            estimates[f"target_{int(target_frac * 100)}pct_threshold"] = 0.0

    return estimates


# =============================================================================
# MAIN ANALYSIS FUNCTION
# =============================================================================


# =============================================================================
# ADAPTIVE THRESHOLD PREDICTION
# =============================================================================


def predict_optimal_threshold(probs: np.ndarray, method: str = "learned") -> float:
    """Predict the optimal threshold for a probability volume based on its characteristics.

    This function uses learned correlations between probability features and optimal
    thresholds to select an appropriate threshold for each sample.

    Updated for Dataset320_VesuviusSurface model (2026-02-02):
    Analysis of 80 samples showed:
    - Best fixed threshold: 0.20 (mean score: 0.6683)
    - Adaptive selection: 0.6695 (+0.0012 gain)
    - Oracle (best per sample): 0.6756

    Key findings from 80 sample analysis:
    - Threshold distribution: 0.25 optimal for 35, 0.20 for 19, 0.15 for 14, 0.10 for 12
    - High kurtosis (>4.5) + low prob_mean (<0.09) -> use 0.10
    - Moderate kurtosis (>4.0) + low prob_mean (<0.10) -> use 0.15
    - Otherwise -> use 0.20 (best average performance)

    Args:
        probs: 3D array of foreground probabilities (values 0-1)
        method: Prediction method - "learned", "regression", "rule_based", or "hybrid"

    Returns:
        Predicted optimal threshold (between 0.10 and 0.25)

    """
    flat_probs = probs.flatten()

    # Compute key features
    prob_mean = np.mean(flat_probs)
    prob_std = np.std(flat_probs)
    prob_kurtosis_val = kurtosis(flat_probs)
    prob_skewness_val = skew(flat_probs)

    # Thresholded volume fractions
    frac_above_50 = (flat_probs > 0.5).mean()
    frac_above_40 = (flat_probs > 0.4).mean()

    # High confidence fraction
    frac_high_conf = (flat_probs > 0.7).mean()

    # Bimodality and decisive ratio
    # Bimodality coefficient: (skewness^2 + 1) / kurtosis
    bimodality_coef = (
        (prob_skewness_val**2 + 1) / (prob_kurtosis_val + 3) if (prob_kurtosis_val + 3) > 0 else 0.5
    )

    # Decisive ratio: fraction of positive predictions that are high confidence
    positive_mask = flat_probs > 0.15
    if positive_mask.sum() > 0:
        decisive_ratio = (flat_probs[positive_mask] > 0.7).sum() / positive_mask.sum()
    else:
        decisive_ratio = 0.0

    if method == "learned":
        # Use learned decision boundaries from 80 sample analysis (2026-02-02)
        # Key insight: 0.20 is best on average, but some samples benefit from lower thresholds
        #
        # Samples benefiting from 0.10 threshold:
        # - High kurtosis (peaked distribution): >4.5
        # - Low probability mean: <0.09
        # - This captures samples where model is confident but predictions are sparse
        #
        # Samples benefiting from 0.15 threshold:
        # - Moderate kurtosis: >4.0
        # - Low-moderate probability mean: <0.10
        #
        # All other samples: use 0.20 (best average performance)

        if prob_kurtosis_val > 4.5 and prob_mean < 0.09:
            # Very peaked distribution with low mean -> use aggressive threshold
            # 15 samples in analysis matched this, avg gain +0.0065
            threshold = 0.10
        elif prob_kurtosis_val > 4.0 and prob_mean < 0.10:
            # Peaked distribution with moderate mean -> use medium threshold
            # 7 samples in analysis matched this, avg gain +0.0001
            threshold = 0.15
        else:
            # Default to best average threshold
            # 58 samples in analysis, consistent with best fixed threshold
            threshold = 0.20

    elif method == "regression":
        # Linear regression based on key features (updated 2026-02-02)
        # Normalization parameters from 80 sample analysis:
        # prob_mean: mean=0.120, std=0.029
        # prob_kurtosis: mean=4.08, std=1.97
        z_prob_mean = (prob_mean - 0.120) / 0.029
        z_kurtosis = (prob_kurtosis_val - 4.08) / 1.97

        # Regression coefficients from analysis (weak correlations, R^2 ~ 0.05)
        # prob_p99: +0.030, threshold_sensitivity: -0.025, max_local_variance: +0.019
        # Use simplified model focused on strongest predictors
        threshold = 0.20  # Start at best average
        threshold -= 0.015 * z_kurtosis  # Higher kurtosis -> lower threshold
        threshold -= 0.010 * z_prob_mean  # Lower prob_mean -> lower threshold

    elif method == "rule_based":
        # Simple rule-based system from 80 sample analysis (2026-02-02)
        # Conservative: only deviate from 0.20 when strong signal

        if (prob_kurtosis_val > 5.5 and prob_mean < 0.10) or (
            prob_kurtosis_val > 4.5 and prob_mean < 0.09
        ):
            threshold = 0.10
        elif prob_mean > 0.15 and frac_high_conf > 0.07:
            # High mean with high confidence predictions -> can use higher threshold
            threshold = 0.25
        else:
            threshold = 0.20

    elif method == "hybrid":
        # Combine regression with rule-based constraints (updated 2026-02-02)

        # Start with regression baseline
        z_prob_mean = (prob_mean - 0.120) / 0.029
        z_kurtosis = (prob_kurtosis_val - 4.08) / 1.97

        threshold = 0.20
        threshold -= 0.015 * z_kurtosis
        threshold -= 0.010 * z_prob_mean

        # Apply rule-based boundaries based on strong signals
        if prob_kurtosis_val > 4.5 and prob_mean < 0.09:
            threshold = min(threshold, 0.10)
        elif prob_kurtosis_val > 4.0 and prob_mean < 0.10:
            threshold = min(threshold, 0.15)

    else:
        # Default to best fixed threshold from analysis
        threshold = 0.20

    # Clamp to valid range [0.10, 0.25]
    threshold = np.clip(threshold, 0.10, 0.25)

    return float(threshold)


def predict_threshold_for_file(pred_path: Path, method: str = "learned") -> tuple[float, dict]:
    """Predict optimal threshold for a prediction file.

    Args:
        pred_path: Path to NPZ prediction file
        method: Prediction method ("learned", "regression", "rule_based", "hybrid")

    Returns:
        Tuple of (predicted_threshold, feature_dict)

    """
    data = np.load(pred_path)
    if "probabilities" not in data:
        raise ValueError(f"No probabilities key in {pred_path}")

    probs_full = data["probabilities"]

    # Handle shape: (2, D, H, W) where channel 1 is foreground
    if probs_full.ndim == 4 and probs_full.shape[0] == 2:
        probs = probs_full[1]
    else:
        probs = probs_full

    threshold = predict_optimal_threshold(probs, method)

    # Also return key features for transparency
    flat_probs = probs.flatten()
    features = {
        "prob_mean": float(np.mean(flat_probs)),
        "prob_std": float(np.std(flat_probs)),
        "prob_kurtosis": float(kurtosis(flat_probs)),
        "prob_skewness": float(skew(flat_probs)),
        "frac_high_conf": float((flat_probs > 0.7).mean()),
        "frac_above_50": float((flat_probs > 0.5).mean()),
    }

    return threshold, features


def analyze_prediction_comprehensive(pred_path: Path) -> dict:
    """Comprehensively analyze a single prediction file.

    Args:
        pred_path: Path to NPZ prediction file

    Returns:
        Dictionary with all analysis results

    """
    try:
        data = np.load(pred_path)
        if "probabilities" not in data:
            return {"filename": pred_path.stem, "error": "No probabilities key"}

        probs_full = data["probabilities"]

        # Handle shape: (2, D, H, W) where channel 1 is foreground
        if probs_full.ndim == 4 and probs_full.shape[0] == 2:
            probs = probs_full[1]  # Foreground channel
        else:
            probs = probs_full

    except Exception as e:
        return {"filename": pred_path.stem, "error": str(e)}

    stats = {
        "filename": pred_path.stem,
        "shape_z": probs.shape[0],
        "shape_y": probs.shape[1],
        "shape_x": probs.shape[2],
        "total_voxels": int(np.prod(probs.shape)),
    }

    # 1. Probability distribution analysis
    stats.update(analyze_probability_distribution(probs))

    # 2. Bimodality analysis
    stats.update(analyze_bimodality(probs))

    # 3. Thresholded volume analysis
    stats.update(analyze_thresholded_volumes(probs))

    # 4. Spatial coherence analysis
    stats.update(analyze_spatial_coherence(probs))

    # 5. Z-continuity analysis at multiple thresholds
    for thresh in [0.10, 0.15, 0.20]:
        z_stats = analyze_z_continuity(probs, threshold=thresh)
        for key, value in z_stats.items():
            stats[f"t{int(thresh * 100)}_{key}"] = value

    # 6. Border connectivity
    border_stats = analyze_border_connectivity(probs, threshold=0.15)
    for key, value in border_stats.items():
        stats[f"border_{key}"] = value

    # 7. Confidence region analysis
    stats.update(analyze_confidence_regions(probs))

    # 8. Component analysis at key thresholds
    for thresh in [0.08, 0.15, 0.20]:
        comp_stats = analyze_components_at_threshold(probs, threshold=thresh)
        for key, value in comp_stats.items():
            stats[f"comp_t{int(thresh * 100)}_{key}"] = value

    # 9. Threshold estimation heuristics
    stats.update(estimate_optimal_threshold(probs))

    return stats


def load_scoring_results(csv_path: Path) -> pd.DataFrame:
    """Load scoring results from CSV."""
    df = pd.read_csv(csv_path)
    return df


def compute_optimal_thresholds(scores_df: pd.DataFrame) -> pd.DataFrame:
    """Compute optimal threshold per sample from scoring results.

    Args:
        scores_df: DataFrame with scoring results

    Returns:
        DataFrame with per-sample optimal threshold info

    """
    # Group by filename and find best threshold
    threshold_map = {
        "thresh_0.08": 0.08,
        "thresh_0.15": 0.15,
        "thresh_0.20": 0.20,
    }

    results = []

    for filename, group in scores_df.groupby("filename"):
        best_row = group.loc[group["final"].idxmax()]
        best_method = best_row["postprocess_method"]
        best_score = best_row["final"]

        # Extract threshold value
        if best_method in threshold_map:
            best_threshold = threshold_map[best_method]
        else:
            # Try to extract from method name
            import re

            match = re.search(r"(\d+\.\d+)", best_method)
            best_threshold = float(match.group(1)) if match else 0.15

        # Get all scores for this sample
        scores_by_thresh = {}
        for _, row in group.iterrows():
            method = row["postprocess_method"]
            if method in threshold_map:
                scores_by_thresh[threshold_map[method]] = row["final"]

        results.append(
            {
                "filename": filename,
                "optimal_threshold": best_threshold,
                "optimal_score": best_score,
                "score_at_0.08": scores_by_thresh.get(0.08, np.nan),
                "score_at_0.15": scores_by_thresh.get(0.15, np.nan),
                "score_at_0.20": scores_by_thresh.get(0.20, np.nan),
            },
        )

    return pd.DataFrame(results)


def correlate_features_with_threshold(
    features_df: pd.DataFrame,
    optimal_df: pd.DataFrame,
) -> pd.DataFrame:
    """Correlate prediction features with optimal threshold choice.

    Args:
        features_df: DataFrame with prediction features
        optimal_df: DataFrame with optimal thresholds per sample

    Returns:
        DataFrame with correlation analysis

    """
    # Ensure filename columns are same type (string)
    features_df = features_df.copy()
    optimal_df = optimal_df.copy()
    features_df["filename"] = features_df["filename"].astype(str)
    optimal_df["filename"] = optimal_df["filename"].astype(str)

    # Merge on filename
    merged = features_df.merge(optimal_df, on="filename")

    # Compute correlations with optimal threshold
    numeric_cols = features_df.select_dtypes(include=[np.number]).columns
    numeric_cols = [c for c in numeric_cols if c != "filename"]

    correlations = []
    for col in numeric_cols:
        if col in merged.columns:
            corr = merged[col].corr(merged["optimal_threshold"])
            correlations.append(
                {
                    "feature": col,
                    "correlation_with_optimal_thresh": corr,
                    "abs_correlation": abs(corr),
                },
            )

    corr_df = pd.DataFrame(correlations)
    corr_df = corr_df.sort_values("abs_correlation", ascending=False)

    return corr_df, merged


def run_analysis(
    pred_dir: Path,
    output_dir: Path = None,
    scores_csv: Path = None,
    num_workers: int = None,
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """Run analysis on all prediction files.

    Args:
        pred_dir: Directory containing NPZ prediction files
        output_dir: Directory for output files (default: pred_dir)
        scores_csv: Optional path to scoring results CSV
        num_workers: Number of parallel workers

    Returns:
        Tuple of (features_df, correlation_df or None)

    """
    if output_dir is None:
        output_dir = pred_dir

    if num_workers is None:
        num_workers = min(cpu_count(), 8)

    # Get all NPZ files
    pred_files = sorted(pred_dir.glob("*.npz"))
    print(f"Found {len(pred_files)} prediction files")
    print(f"Using {num_workers} workers")

    # Process in parallel
    all_stats = []
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(analyze_prediction_comprehensive, f): f for f in pred_files}

        for future in tqdm(futures, desc="Analyzing predictions"):
            try:
                stats = future.result()
                all_stats.append(stats)
            except Exception as e:
                print(f"Error processing {futures[future]}: {e}")

    features_df = pd.DataFrame(all_stats)

    # Save features
    output_path = output_dir / "prediction_features.csv"
    features_df.to_csv(output_path, index=False)
    print(f"Saved features to: {output_path}")

    # Correlation analysis if scoring results provided
    corr_df = None
    merged_df = None

    if scores_csv and scores_csv.exists():
        print("\nLoading scoring results for correlation analysis...")
        scores_df = load_scoring_results(scores_csv)
        optimal_df = compute_optimal_thresholds(scores_df)

        # Save optimal thresholds
        optimal_path = output_dir / "optimal_thresholds.csv"
        optimal_df.to_csv(optimal_path, index=False)
        print(f"Saved optimal thresholds to: {optimal_path}")

        # Compute correlations
        corr_df, merged_df = correlate_features_with_threshold(features_df, optimal_df)

        # Save correlations
        corr_path = output_dir / "feature_threshold_correlations.csv"
        corr_df.to_csv(corr_path, index=False)
        print(f"Saved correlations to: {corr_path}")

        # Save merged data for further analysis
        merged_path = output_dir / "features_with_scores.csv"
        merged_df.to_csv(merged_path, index=False)
        print(f"Saved merged data to: {merged_path}")

        # Print top correlations
        print("\n" + "=" * 60)
        print("TOP FEATURES CORRELATED WITH OPTIMAL THRESHOLD")
        print("=" * 60)
        print(corr_df.head(20).to_string(index=False))

    return features_df, corr_df, merged_df


def print_analysis_summary(features_df: pd.DataFrame):
    """Print summary statistics of the prediction features."""
    print("\n" + "=" * 60)
    print("PREDICTION FEATURE SUMMARY")
    print("=" * 60)

    print(f"\nTotal predictions analyzed: {len(features_df)}")

    # Key distribution metrics
    print("\n--- PROBABILITY DISTRIBUTION ---")
    for col in ["prob_mean", "prob_std", "prob_median", "prob_p95"]:
        if col in features_df.columns:
            print(f"{col}: {features_df[col].mean():.4f} ± {features_df[col].std():.4f}")

    # Bimodality
    print("\n--- BIMODALITY METRICS ---")
    for col in ["bimodality_coef", "valley_depth", "fraction_high_prob"]:
        if col in features_df.columns:
            print(f"{col}: {features_df[col].mean():.4f} ± {features_df[col].std():.4f}")

    # Confidence
    print("\n--- CONFIDENCE METRICS ---")
    for col in ["decisive_ratio", "confident_to_uncertain_ratio"]:
        if col in features_df.columns:
            print(f"{col}: {features_df[col].mean():.4f} ± {features_df[col].std():.4f}")

    # Spatial
    print("\n--- SPATIAL COHERENCE ---")
    for col in ["mean_gradient_magnitude", "edge_sharpness"]:
        if col in features_df.columns:
            print(f"{col}: {features_df[col].mean():.4f} ± {features_df[col].std():.4f}")

    # Threshold estimates
    print("\n--- THRESHOLD ESTIMATES ---")
    for col in ["otsu_threshold", "target_10pct_threshold"]:
        if col in features_df.columns:
            print(f"{col}: {features_df[col].mean():.4f} ± {features_df[col].std():.4f}")


if __name__ == "__main__":
    # Configuration
    BASE_DIR = Path(os.getenv("BASE_DIR", "."))  # repo root; override with the BASE_DIR env var
    PRED_DIR = (
        BASE_DIR
        / "scoring/temp_results"
        / "nnUNetTrainer_FineTune500epochs_LR001__nnUNetResEncUNetLPlans__3d_fullres"
        / "predictions"
    )
    OUTPUT_DIR = BASE_DIR / "scoring/temp_results"
    # Use comprehensive_results (per-file scores), not comprehensive_summary (aggregated stats)
    SCORES_CSV = BASE_DIR / "scoring/temp_results/comprehensive_results_20260201_123122.csv"
    print("=" * 70)
    print("PREDICTION PROBABILITY ANALYSIS")
    print("=" * 70)
    print(f"Prediction directory: {PRED_DIR}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Scores CSV: {SCORES_CSV}")

    # Run analysis
    features_df, corr_df, merged_df = run_analysis(
        pred_dir=PRED_DIR,
        output_dir=OUTPUT_DIR,
        scores_csv=SCORES_CSV,
        num_workers=8,
    )

    # Print summary
    print_analysis_summary(features_df)

    # Additional analysis: which features best predict threshold choice
    if corr_df is not None and merged_df is not None:
        print("\n" + "=" * 70)
        print("THRESHOLD PREDICTION INSIGHTS")
        print("=" * 70)

        # Group samples by optimal threshold
        for thresh in [0.08, 0.15, 0.20]:
            subset = merged_df[merged_df["optimal_threshold"] == thresh]
            if len(subset) > 0:
                print(f"\nSamples optimal at threshold {thresh}: {len(subset)}")
                # Show key distinguishing features
                for feat in ["prob_mean", "decisive_ratio", "bimodality_coef", "otsu_threshold"]:
                    if feat in subset.columns:
                        print(f"  {feat}: {subset[feat].mean():.4f} ± {subset[feat].std():.4f}")

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)
