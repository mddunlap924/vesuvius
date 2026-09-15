"""Enhanced scoring script with advanced post-processing techniques.

Compares multiple post-processing pipelines against baseline threshold
to find optimal strategies for thin ribbon segmentation.
"""

import glob
import logging
import multiprocessing
import os
import sys
from concurrent.futures import BrokenExecutor, ProcessPoolExecutor, as_completed
from datetime import datetime
from functools import partial
from multiprocessing import cpu_count
from pathlib import Path
from typing import TypedDict

import numpy as np
import pandas as pd

# GPU imports for MONAI Frangi
import torch

# Import topology-focused adaptive post-processing (2026-02-03)
from adaptive_topo_postprocess import (
    adaptive_topo_postprocess,
    apply_targeted_topo_enhancement,
    extract_topo_features,
    get_topo_optimized_threshold,
    is_topo_at_risk,
    recommend_strategy,
)

# Import adaptive thresholding module
from analyze_prediction_probabilities import (
    adaptive_hysteresis_postprocess,
    adaptive_threshold_postprocess,
    get_adaptive_threshold,
    predict_topo_optimal_threshold,
)

# Import micro-hole filling post-processing (2026-02-09)
from microhole_postprocess import (
    adaptive_border_aware_holefill,
    adaptive_border_aware_holefill_v2,
    adaptive_border_aware_holefill_v3,
)
from PIL import Image, ImageSequence

# Import GT-informed post-processing functions
from post_processing_probs import (
    GT_PARAMS,
    cc3d_contact_analysis,
    cc3d_continuous_threshold,
    cc3d_dust_filter,
    cc3d_iterative_refinement,
    cc3d_largest_k,
    cc3d_per_component_processing,
    cc3d_shape_filter,
    cc3d_skeleton_bridge_gaps,
    cc3d_statistics_filter,
    cc3d_watershed_separation,
    # Component filtering functions
    filter_by_border_endpoints,
    filter_by_path_smoothness,
    hysteresis_gt_informed,
    postprocess_adaptive_threshold,
    # Border-aware methods
    postprocess_border_aware,
    postprocess_border_aware_hysteresis,
    postprocess_border_extend,
    postprocess_border_path_aware,
    postprocess_border_strict,
    postprocess_cc3d_advanced,
    postprocess_cc3d_full,
    postprocess_cc3d_iterative,
    postprocess_cc3d_largest,
    # NEW: CC3D-based methods
    postprocess_cc3d_topology,
    postprocess_cc3d_topology_strict,
    postprocess_full_pipeline,
    postprocess_gt,
    postprocess_gt_balanced,
    postprocess_gt_comprehensive,
    postprocess_gt_minimal,
    postprocess_gt_strict,
    postprocess_gt_surface_optimized,
    postprocess_gt_topology,
    postprocess_gt_voi_optimized,
    threshold_gt_optimized,
)

# Import post-processing module
from postprocess_ribbons import (
    PIPELINES,
    apply_pipeline,
    enforce_border_connectivity,
    enforce_z_continuity,
    ensemble_thresholds,
    fill_holes_2d_per_slice,
    fill_holes_3d,
    frangi_surfaceness_filter,
    hysteresis_threshold,
    morphological_cleanup,
    remove_small_components,
    simple_threshold,
    surfaceness_threshold,
)

# Import precision-focused post-processing (2026-02-05)
from precision_postprocess import (
    combined_precision_postprocess,
    highconf_seed_postprocess,
    multiscale_consensus_postprocess,
    opening_cleanup_postprocess,
    precision_adaptive_postprocess,
    precision_border_aware_postprocess,
    volume_calibrated_postprocess,
)
from scipy.stats import rankdata
from skimage.filters import frangi
from topometrics.leaderboard import compute_leaderboard_score

GPU_AVAILABLE = torch.cuda.is_available()


# Global cache for precomputed Frangi results
# Key: (method_name, filename) -> binary mask (np.ndarray)
_frangi_cache: dict[tuple[str, str], np.ndarray] = {}


def precompute_gpu_frangi_batch(
    paired_predictions: list,
    methods_config: dict,
    load_volume_fn,
) -> dict[tuple[str, str], np.ndarray]:
    """Precompute GPU Frangi filter for all samples and GPU methods at once.

    This batches GPU operations to maximize utilization by:
    1. Creating the FrangiFilter once per method config
    2. Processing all samples through GPU before moving to scoring

    Args:
        paired_predictions: List of prediction path pairs with weights
        methods_config: Dict of method_name -> params (only gpu_* methods processed)
        load_volume_fn: Function to load NPZ volume

    Returns:
        Dict mapping (method_name, filename) -> binary mask

    """
    global _frangi_cache
    _frangi_cache.clear()

    if not GPU_AVAILABLE:
        print("WARNING: GPU not available, skipping Frangi precomputation")
        return _frangi_cache

    device = torch.device("cuda")

    # Filter to only GPU methods
    gpu_methods = {k: v for k, v in methods_config.items() if k.startswith("gpu_frangi")}

    if not gpu_methods:
        return _frangi_cache

    print(f"\n{'=' * 60}")
    print("PRECOMPUTING GPU FRANGI FILTERS")
    print(f"{'=' * 60}")
    print(f"Methods: {list(gpu_methods.keys())}")
    print(f"Samples: {len(paired_predictions)}")

    # Group methods by their Frangi parameters (sigmas, alpha, beta, gamma, black_ridges)
    # to reuse filter instances
    filter_cache: dict[tuple, FrangiFilter] = {}

    for method_name, params in gpu_methods.items():
        sigmas = params.get("sigmas", (1, 3, 5, 7))
        alpha = params.get("alpha", 0.5)
        beta = params.get("beta", 0.5)
        gamma = params.get("gamma", 15.0)
        black_ridges = params.get("black_ridges", False)
        pre_threshold = params.get("pre_threshold")
        post_threshold = params.get("post_threshold", 0.3)
        normalize_output = params.get("normalize_output", True)

        # Create filter key (excluding pre/post threshold which are applied separately)
        filter_key = (tuple(sigmas), alpha, beta, gamma, black_ridges)

        if filter_key not in filter_cache:
            print(
                f"Creating FrangiFilter: sigmas={sigmas}, alpha={alpha}, beta={beta}, gamma={gamma}",
            )
            frangi_filter = FrangiFilter(
                ndim=3,
                sigmas=sigmas,
                alpha=alpha,
                beta=beta,
                gamma=gamma,
                black_ridges=black_ridges,
            ).to(device)
            filter_cache[filter_key] = frangi_filter

        frangi_filter = filter_cache[filter_key]

        print(f"\nProcessing method: {method_name}")
        print(f"  pre_threshold={pre_threshold}, post_threshold={post_threshold}")

        for i, pred_info in enumerate(paired_predictions):
            filename = Path(pred_info[0][0]).stem

            # Load and combine predictions
            pr = None
            total_weight = 0.0
            for pred_path, weight in pred_info:
                pred_volume = load_volume_fn(pred_path)
                if pr is None:
                    pr = pred_volume.astype(np.float32) * weight
                else:
                    pr += pred_volume.astype(np.float32) * weight
                total_weight += weight
            pr = pr / total_weight

            # Convert to tensor: (1, 1, D, H, W)
            pr_tensor = torch.from_numpy(pr).float().unsqueeze(0).unsqueeze(0).to(device)

            # Apply pre-threshold if specified
            if pre_threshold is not None:
                pr_tensor = (pr_tensor > pre_threshold).float()

            # Apply GPU Frangi filter (MONAI native)
            with torch.no_grad():
                surf = frangi_filter(pr_tensor)

            # Normalize if requested
            if normalize_output:
                surf_min = surf.min()
                surf_max = surf.max()
                if surf_max > surf_min:
                    surf = (surf - surf_min) / (surf_max - surf_min)

            # Apply post-threshold and convert to binary mask
            mask = (surf > post_threshold).squeeze().cpu().numpy().astype(np.uint8)

            # Cache the result
            _frangi_cache[(method_name, filename)] = mask

            if (i + 1) % 10 == 0 or i == len(paired_predictions) - 1:
                print(f"  Processed {i + 1}/{len(paired_predictions)} samples")

        # Clear GPU memory after each method
        torch.cuda.empty_cache()

    print(f"\nPrecomputed {len(_frangi_cache)} Frangi results")
    print("GPU memory cleared")

    return _frangi_cache


def get_cached_frangi_result(method_name: str, filename: str) -> np.ndarray | None:
    """Retrieve precomputed Frangi result from cache."""
    return _frangi_cache.get((method_name, filename))


class ModelConfig(TypedDict):
    path: str
    weight: float
    full_path: Path
    preds: list[Path]


class PredDirConfig(TypedDict):
    base_dir: Path
    model0: ModelConfig
    model1: ModelConfig


# =============================================================================
# ENSEMBLE COMBINATION STRATEGIES
# =============================================================================


def ensemble_weighted_average(volumes: list[np.ndarray], weights: list[float]) -> np.ndarray:
    """Standard weighted average of probability maps."""
    result = np.zeros_like(volumes[0], dtype=np.float32)
    total_weight = sum(weights)
    for vol, w in zip(volumes, weights):
        result += vol.astype(np.float32) * w
    return result / total_weight


def ensemble_geometric_mean(volumes: list[np.ndarray], weights: list[float]) -> np.ndarray:
    """Geometric mean - emphasizes agreement between models.

    Areas where both models predict high get boosted,
    areas where one predicts low get suppressed.
    """
    # Add small epsilon to avoid log(0)
    eps = 1e-7
    result = np.zeros_like(volumes[0], dtype=np.float32)
    total_weight = sum(weights)

    for vol, w in zip(volumes, weights):
        result += w * np.log(vol.astype(np.float32) + eps)

    return np.exp(result / total_weight) - eps


def ensemble_harmonic_mean(volumes: list[np.ndarray], weights: list[float]) -> np.ndarray:
    """Harmonic mean - heavily penalizes disagreement.

    Even more conservative than geometric mean.
    """
    eps = 1e-7
    result = np.zeros_like(volumes[0], dtype=np.float32)
    total_weight = sum(weights)

    for vol, w in zip(volumes, weights):
        result += w / (vol.astype(np.float32) + eps)

    return total_weight / (result + eps)


def ensemble_maximum(volumes: list[np.ndarray], weights: list[float]) -> np.ndarray:
    """Maximum (union-like) - keeps highest probability at each voxel.

    Good for capturing all potential ribbon regions.
    """
    # Weights are ignored for max, but we still accept them for API consistency
    return np.maximum.reduce([v.astype(np.float32) for v in volumes])


def ensemble_minimum(volumes: list[np.ndarray], weights: list[float]) -> np.ndarray:
    """Minimum (intersection-like) - keeps lowest probability at each voxel.

    Very conservative, only keeps regions both models agree on.
    """
    return np.minimum.reduce([v.astype(np.float32) for v in volumes])


def ensemble_soft_or(volumes: list[np.ndarray], weights: list[float]) -> np.ndarray:
    """Soft OR combination: P(A or B) = P(A) + P(B) - P(A)*P(B).

    Probabilistic union that avoids double-counting.
    """
    if len(volumes) == 1:
        return volumes[0].astype(np.float32)

    result = volumes[0].astype(np.float32)
    for vol in volumes[1:]:
        v = vol.astype(np.float32)
        result = result + v - (result * v)
    return result


def ensemble_soft_and(volumes: list[np.ndarray], weights: list[float]) -> np.ndarray:
    """Soft AND combination: P(A and B) = P(A) * P(B).

    Probabilistic intersection - very conservative.
    """
    result = np.ones_like(volumes[0], dtype=np.float32)
    for vol in volumes:
        result *= vol.astype(np.float32)
    return result


def ensemble_power_mean(
    volumes: list[np.ndarray],
    weights: list[float],
    p: float = 2.0,
) -> np.ndarray:
    """Generalized power mean with parameter p.

    p = -1: harmonic mean
    p = 0: geometric mean (limit)
    p = 1: arithmetic mean
    p = 2: quadratic mean (RMS)
    p -> inf: maximum
    p -> -inf: minimum

    Higher p emphasizes higher values, lower p emphasizes lower values.
    """
    eps = 1e-7
    total_weight = sum(weights)

    if abs(p) < 0.01:  # Approximate geometric mean
        return ensemble_geometric_mean(volumes, weights)

    result = np.zeros_like(volumes[0], dtype=np.float32)
    for vol, w in zip(volumes, weights):
        result += w * np.power(vol.astype(np.float32) + eps, p)

    return np.power(result / total_weight, 1.0 / p) - eps


def ensemble_adaptive_threshold(
    volumes: list[np.ndarray],
    weights: list[float],
    agreement_boost: float = 1.2,
    disagreement_penalty: float = 0.8,
) -> np.ndarray:
    """Adaptive weighting based on model agreement.

    Boosts regions where models agree (both high or both low),
    penalizes regions where they disagree.
    """
    if len(volumes) < 2:
        return volumes[0].astype(np.float32)

    v0 = volumes[0].astype(np.float32)
    v1 = volumes[1].astype(np.float32)

    # Measure agreement (1 when same, 0 when opposite)
    agreement = 1.0 - np.abs(v0 - v1)

    # Base weighted average
    base = ensemble_weighted_average(volumes, weights)

    # Boost high-confidence agreements, penalize disagreements
    # Where both are high and agree -> boost
    # Where they disagree -> reduce
    high_agreement = (agreement > 0.7) & (base > 0.3)
    low_agreement = agreement < 0.3

    result = base.copy()
    result[high_agreement] *= agreement_boost
    result[low_agreement] *= disagreement_penalty

    return np.clip(result, 0, 1)


def ensemble_rank_average(volumes: list[np.ndarray], weights: list[float]) -> np.ndarray:
    """Rank-based averaging - robust to different probability scales.

    Converts probabilities to ranks, averages ranks, converts back.
    """
    shape = volumes[0].shape
    flat_vols = [v.flatten().astype(np.float32) for v in volumes]

    # Convert to percentile ranks
    ranked = []
    for vol in flat_vols:
        ranks = rankdata(vol, method="average")
        percentile_ranks = ranks / len(ranks)
        ranked.append(percentile_ranks)

    # Weighted average of ranks
    total_weight = sum(weights)
    result = np.zeros_like(ranked[0])
    for r, w in zip(ranked, weights):
        result += r * w
    result /= total_weight

    return result.reshape(shape)


def ensemble_confidence_weighted(
    volumes: list[np.ndarray],
    weights: list[float],
    confidence_power: float = 2.0,
) -> np.ndarray:
    """Weight by prediction confidence (distance from 0.5).

    Gives more weight to confident predictions.
    """
    total_weight = sum(weights)

    confidences = []
    for vol in volumes:
        v = vol.astype(np.float32)
        # Confidence is distance from 0.5, scaled to [0, 1]
        conf = np.abs(v - 0.5) * 2.0
        confidences.append(np.power(conf, confidence_power))

    # Normalize weights per voxel
    weighted_sum = np.zeros_like(volumes[0], dtype=np.float32)
    weight_sum = np.zeros_like(volumes[0], dtype=np.float32)

    for vol, w, conf in zip(volumes, weights, confidences):
        v = vol.astype(np.float32)
        voxel_weight = w * conf
        weighted_sum += v * voxel_weight
        weight_sum += voxel_weight

    # Avoid division by zero
    weight_sum = np.maximum(weight_sum, 1e-7)
    return weighted_sum / weight_sum


def ensemble_stacked_boost(
    volumes: list[np.ndarray],
    weights: list[float],
    high_thresh: float = 0.5,
    low_thresh: float = 0.2,
    boost_factor: float = 1.3,
) -> np.ndarray:
    """Stacked boosting - use strong predictions to boost weak ones.

    If one model is confident, boost the other model's uncertain predictions.
    """
    if len(volumes) < 2:
        return volumes[0].astype(np.float32)

    v0 = volumes[0].astype(np.float32)
    v1 = volumes[1].astype(np.float32)

    # Start with weighted average
    result = ensemble_weighted_average(volumes, weights)

    # Where model0 is confident positive and model1 is uncertain, boost
    boost_mask_0 = (v0 > high_thresh) & (v1 > low_thresh) & (v1 < high_thresh)
    result[boost_mask_0] = np.minimum(result[boost_mask_0] * boost_factor, 1.0)

    # Where model1 is confident positive and model0 is uncertain, boost
    boost_mask_1 = (v1 > high_thresh) & (v0 > low_thresh) & (v0 < high_thresh)
    result[boost_mask_1] = np.minimum(result[boost_mask_1] * boost_factor, 1.0)

    return result


# Registry of ensemble methods
ENSEMBLE_METHODS = {
    "weighted_average": ensemble_weighted_average,
    "geometric_mean": ensemble_geometric_mean,
    "harmonic_mean": ensemble_harmonic_mean,
    "maximum": ensemble_maximum,
    "minimum": ensemble_minimum,
    "soft_or": ensemble_soft_or,
    "soft_and": ensemble_soft_and,
    "rank_average": ensemble_rank_average,
    "confidence_weighted": ensemble_confidence_weighted,
    "adaptive": ensemble_adaptive_threshold,
    "stacked_boost": ensemble_stacked_boost,
}


def combine_predictions(
    pred_paths_with_weights: list[tuple[Path, float]],
    load_fn,
    ensemble_method: str = "weighted_average",
    ensemble_params: dict | None = None,
) -> np.ndarray:
    """Combine multiple model predictions using specified ensemble method.

    Args:
        pred_paths_with_weights: List of (path, weight) tuples
        load_fn: Function to load volume from path
        ensemble_method: Name of ensemble method to use
        ensemble_params: Additional parameters for ensemble method

    Returns:
        Combined probability map (float32, 0-1)

    """
    if ensemble_params is None:
        ensemble_params = {}

    # Load all volumes
    volumes = []
    weights = []
    for pred_path, weight in pred_paths_with_weights:
        if weight > 0:  # Only load models with non-zero weight
            volumes.append(load_fn(pred_path))
            weights.append(weight)

    # Single model case
    if len(volumes) == 1:
        return volumes[0].astype(np.float32)

    # Get ensemble function
    if ensemble_method not in ENSEMBLE_METHODS:
        raise ValueError(
            f"Unknown ensemble method: {ensemble_method}. "
            f"Available: {list(ENSEMBLE_METHODS.keys())}",
        )

    ensemble_fn = ENSEMBLE_METHODS[ensemble_method]

    # Handle methods with extra parameters
    if ensemble_method == "adaptive":
        return ensemble_fn(
            volumes,
            weights,
            agreement_boost=ensemble_params.get("agreement_boost", 1.2),
            disagreement_penalty=ensemble_params.get("disagreement_penalty", 0.8),
        )
    if ensemble_method == "confidence_weighted":
        return ensemble_fn(
            volumes,
            weights,
            confidence_power=ensemble_params.get("confidence_power", 2.0),
        )
    if ensemble_method == "stacked_boost":
        return ensemble_fn(
            volumes,
            weights,
            high_thresh=ensemble_params.get("high_thresh", 0.5),
            low_thresh=ensemble_params.get("low_thresh", 0.2),
            boost_factor=ensemble_params.get("boost_factor", 1.3),
        )
    if ensemble_method == "power_mean":
        return ensemble_power_mean(
            volumes,
            weights,
            p=ensemble_params.get("p", 2.0),
        )
    return ensemble_fn(volumes, weights)


def load_volume(path):
    im = Image.open(path)
    slices = []
    for i, page in enumerate(ImageSequence.Iterator(im)):
        slice_array = np.array(page)
        slices.append(slice_array)
    volume = np.stack(slices, axis=0)
    return volume


def load_npz_volume(path):
    data = np.load(path)
    if "probabilities" in data:
        return data["probabilities"][1]
    raise ValueError(f"NPZ file {path} does not contain 'probabilities' key.")


def load_npy_volume(path):
    return np.load(path)


def postprocess_prediction(
    pr: np.ndarray,
    method: str = "threshold",
    params: dict | None = None,
    filename: str | None = None,
) -> np.ndarray:
    """Apply post-processing to probability map.

    Args:
        pr: Probability map (float32, 0-1)
        method: Post-processing method name
        params: Parameters for the method
        filename: Sample filename (used for GPU Frangi cache lookup)

    Returns:
        Binary mask (uint8)

    """
    if params is None:
        params = {}

    # Handle GPU Frangi surfaceness (MONAI-based)
    # Use precomputed cache if available
    if method.startswith("gpu_frangi"):
        if filename:
            cached = get_cached_frangi_result(method, filename)
            if cached is not None:
                return cached
        # Fallback: should not happen if precompute was called
        raise RuntimeError(
            f"GPU Frangi result not found in cache for method={method}, filename={filename}. "
            "Ensure precompute_gpu_frangi_batch() was called first.",
        )

    # Handle surfaceness_enhanced directly (not via pipeline) - CPU fallback
    # Matches "surfaceness_enhanced", "surfaceness_enhanced_low", etc.
    if method.startswith("surfaceness_enhanced"):
        # Apply Frangi filter to enhance sheet-like structures
        threshold = params.get("threshold", 0.5)
        sigmas = params.get("sigmas", range(1, 10, 2))
        alpha = params.get("alpha", 0.5)
        beta = params.get("beta", 0.5)

        # Apply Frangi filter to 3D volume
        frangi_response = frangi(
            pr,
            sigmas=sigmas,
            alpha=alpha,
            beta=beta,
            black_ridges=False,  # We want bright sheet-like structures
            mode="reflect",
            cval=0,
        )

        # Threshold the Frangi response
        mask = (frangi_response > threshold).astype(np.uint8)
        return mask

    # Check if it's a predefined pipeline first
    if method in PIPELINES:
        return apply_pipeline(pr, method, params.get("custom_params"))

    # Use prefix matching for method names (e.g., "threshold_0.15" -> "threshold")
    if method.startswith("threshold") and "_holes" not in method and "_gt" not in method:
        threshold = params.get("threshold", 0.2)
        return simple_threshold(pr, threshold)

    if method.startswith("hysteresis") and "_gt_cleanup" not in method:
        return hysteresis_threshold(
            pr,
            low_threshold=params.get("low_threshold", 0.15),
            high_threshold=params.get("high_threshold", 0.35),
        )

    if method.startswith("ensemble"):
        return ensemble_thresholds(
            pr,
            thresholds=params.get("thresholds", [0.1, 0.15, 0.2, 0.25, 0.3]),
            voting=params.get("voting", "majority"),
        )

    if method == "threshold_holes":
        # Threshold + hole filling (minimal)
        threshold = params.get("threshold", 0.18)
        mask = simple_threshold(pr, threshold)
        mask = fill_holes_2d_per_slice(mask)
        mask = remove_small_components(mask, min_size=params.get("min_size", 50))
        return mask

    if method == "hysteresis_gt_cleanup":
        # Hysteresis + GT-informed cleanup (no border enforcement)
        mask = hysteresis_threshold(
            pr,
            low_threshold=params.get("low_threshold", 0.10),
            high_threshold=params.get("high_threshold", 0.25),
        )
        mask = fill_holes_2d_per_slice(mask)
        mask = fill_holes_3d(mask, max_hole_size=params.get("max_hole_size", 100))
        mask = remove_small_components(mask, min_size=params.get("min_size", 100))
        mask = enforce_z_continuity(mask, min_z_span=params.get("min_z_span", 10))
        return mask

    # =========================================================================
    # COMBINED: adaptive_border_aware + targeted topo (2026-02-05)
    # Uses adaptive_border_aware for normal samples (best overall approach),
    # but switches to a topo-optimized threshold for at-risk samples
    # (low decisive_ratio or high fragmentation) where topo drags score down.
    # =========================================================================
    if method == "adaptive_border_topo" or method.startswith("adaptive_border_topo_"):
        at_risk, features = is_topo_at_risk(pr)

        if at_risk:
            # At-risk for poor topo: use topo-optimized threshold instead
            opt_threshold = get_topo_optimized_threshold(features)
            mask = simple_threshold(pr, opt_threshold)
            mask = remove_small_components(mask, min_size=params.get("min_size", 500))

            import logging

            logging.info(
                f"adaptive_border_topo: AT_RISK -> topo threshold={opt_threshold:.2f}, "
                f"decisive={features.decisive_ratio:.3f}, comps={features.num_components_t20}",
            )
        else:
            # Normal sample: use adaptive_border_aware (proven best)
            adaptive_method = params.get("adaptive_method", "learned")
            min_size = params.get("min_size", 500)
            mask = adaptive_threshold_postprocess(pr, method=adaptive_method, min_size=min_size)

            border_standoff = params.get("border_standoff", 15)
            require_both = params.get("require_both_endpoints", False)
            min_endpoints = 2 if require_both else 1
            mask = filter_by_border_endpoints(
                mask,
                border_standoff=border_standoff,
                require_start_near_border=require_both,
                require_end_near_border=require_both,
                min_endpoint_near_border=min_endpoints,
            )

        return mask

    # =========================================================================
    # TARGETED TOPO ENHANCEMENT (2026-02-04)
    # Simple, validated approach: only adjust threshold for at-risk samples
    # Validated on 30 samples: +0.0377 improvement, 0 regressions
    # =========================================================================
    if method == "targeted_topo" or method.startswith("targeted_topo_"):
        default_threshold = params.get("default_threshold", 0.20)

        mask, info = apply_targeted_topo_enhancement(
            pr,
            default_threshold=default_threshold,
            return_info=True,
        )

        # Log the decision for analysis
        import logging

        if info["at_risk"]:
            logging.debug(
                f"targeted_topo: AT_RISK sample, strategy={info['strategy']}, "
                f"threshold={info['threshold']}, decisive_ratio={info['decisive_ratio']:.3f}",
            )

        # Optional: Additional border filtering
        if params.get("apply_border_filter", False):
            border_standoff = params.get("border_standoff", 15)
            mask = filter_by_border_endpoints(
                mask,
                border_standoff=border_standoff,
                require_start_near_border=False,
                require_end_near_border=False,
                min_endpoint_near_border=1,
            )

        return mask

    # =========================================================================
    # ADAPTIVE THRESHOLDING METHOD
    # Uses per-sample probability analysis to select optimal threshold
    # =========================================================================

    if method == "adaptive" or method.startswith("adaptive_"):
        # Check for combined adaptive + border_aware method
        if method == "adaptive_border_aware":
            # Step 1: Adaptive thresholding (learned)
            adaptive_method = params.get("adaptive_method", "learned")
            min_size = params.get("min_size", 500)
            mask = adaptive_threshold_postprocess(pr, method=adaptive_method, min_size=min_size)

            # Step 2: Apply border endpoint filtering
            border_standoff = params.get("border_standoff", 15)  # GT_PARAMS default
            require_both = params.get("require_both_endpoints", False)  # basic uses False
            min_endpoints = 2 if require_both else 1
            mask = filter_by_border_endpoints(
                mask,
                border_standoff=border_standoff,
                require_start_near_border=require_both,
                require_end_near_border=require_both,
                min_endpoint_near_border=min_endpoints,
            )
            return mask

        # Check for combined adaptive + hole filling method
        if method == "adaptive_hole_fill":
            # Step 1: Adaptive thresholding (learned)
            adaptive_method = params.get("adaptive_method", "learned")
            min_size = params.get("min_size", 500)
            mask = adaptive_threshold_postprocess(pr, method=adaptive_method, min_size=min_size)

            # Step 2: Fill holes to recover broken ribbon segments
            if params.get("fill_holes_2d", True):
                mask = fill_holes_2d_per_slice(mask)
            if params.get("fill_holes_3d", True):
                max_hole_size = params.get("max_hole_size_3d", 100)
                mask = fill_holes_3d(mask, max_hole_size=max_hole_size)

            # Step 3: Remove small components after hole filling
            mask = remove_small_components(mask, min_size=min_size)
            return mask

        # Check for adaptive + z-continuity method
        if method == "adaptive_z_continuity":
            # Step 1: Adaptive thresholding (learned)
            adaptive_method = params.get("adaptive_method", "learned")
            min_size = params.get("min_size", 500)
            mask = adaptive_threshold_postprocess(pr, method=adaptive_method, min_size=min_size)

            # Step 2: Enforce z-continuity (remove 2D artifacts)
            min_z_span = params.get("min_z_span", 5)
            mask = enforce_z_continuity(mask, min_z_span=min_z_span)
            return mask

        # Check for adaptive hysteresis method
        # Uses learned threshold as HIGH, recovers weak regions with LOW threshold
        if method == "adaptive_hysteresis":
            from analyze_prediction_probabilities import predict_optimal_threshold

            # Get the learned optimal threshold for this sample
            high_thresh = predict_optimal_threshold(pr, method="learned")
            # Use a lower threshold to recover weak regions
            low_thresh = params.get("low_thresh", high_thresh * 0.6)  # 60% of learned
            if "low_thresh_absolute" in params:
                low_thresh = params["low_thresh_absolute"]

            min_size = params.get("min_size", 500)

            # Apply hysteresis thresholding
            mask = hysteresis_threshold(pr, low_threshold=low_thresh, high_threshold=high_thresh)
            mask = remove_small_components(mask, min_size=min_size)
            return mask

        # NEW: Adaptive hysteresis for topology improvement (2026-02-02)
        # Uses probability features to dynamically set hysteresis thresholds
        # Designed to improve topo_dim0 by recovering weak connections
        if method == "adaptive_hysteresis_topo":
            min_size = params.get("min_size", 500)
            use_bridge = params.get("use_bridge_recovery", True)
            mask = adaptive_hysteresis_postprocess(
                pr,
                min_size=min_size,
                use_bridge_recovery=use_bridge,
            )
            return mask

        # NEW: Adaptive hysteresis + border filtering for balanced improvement
        if method == "adaptive_hysteresis_border":
            # Step 1: Apply adaptive hysteresis for topology
            min_size = params.get("min_size", 500)
            use_bridge = params.get("use_bridge_recovery", True)
            mask = adaptive_hysteresis_postprocess(
                pr,
                min_size=min_size,
                use_bridge_recovery=use_bridge,
            )

            # Step 2: Apply border endpoint filtering
            border_standoff = params.get("border_standoff", 15)
            require_both = params.get("require_both_endpoints", False)
            min_endpoints = 2 if require_both else 1
            mask = filter_by_border_endpoints(
                mask,
                border_standoff=border_standoff,
                require_start_near_border=require_both,
                require_end_near_border=require_both,
                min_endpoint_near_border=min_endpoints,
            )
            return mask

        # ==================================================================
        # NEW: TOPOLOGY-FOCUSED ADAPTIVE POST-PROCESSING (2026-02-03)
        # Uses sample-specific strategies based on probability features
        # Strategies derived from 80-sample analysis of topo correlations
        # ==================================================================
        if method == "adaptive_topo_v2" or method.startswith("adaptive_topo_v2_"):
            # Get the recommended strategy and apply
            strategy = params.get("strategy", "auto")

            mask, info = adaptive_topo_postprocess(
                pr,
                strategy=strategy,
                params=params,
                return_info=True,
            )

            # Optional: Additional border filtering
            if params.get("apply_border_filter", False):
                border_standoff = params.get("border_standoff", 15)
                mask = filter_by_border_endpoints(
                    mask,
                    border_standoff=border_standoff,
                    require_start_near_border=False,
                    require_end_near_border=False,
                    min_endpoint_near_border=1,
                )

            return mask

        # ==================================================================
        # ADAPTIVE TOPO V3 (2026-02-03): Refined from 80-sample analysis
        # Uses updated strategy thresholds:
        # - Priority 1: Low decisive_ratio (<0.35) → hysteresis/ultra-cleanup
        # - Priority 2: High fragmentation (>100) → confident/moderate cleanup
        # - Priority 3: High confidence (>0.60) → lower thresholds (0.10-0.15)
        # ==================================================================
        if method == "adaptive_topo_v3" or method.startswith("adaptive_topo_v3_"):
            strategy = params.get("strategy", "auto")

            mask, info = adaptive_topo_postprocess(
                pr,
                strategy=strategy,
                params=params,
                return_info=True,
            )

            # Log the selected strategy for analysis
            import logging

            logging.debug(
                f"adaptive_topo_v3: strategy={info['strategy']}, "
                f"decisive_ratio={info['features']['decisive_ratio']:.3f}, "
                f"components={info['features']['num_components_t20']}",
            )

            # Optional: Additional border filtering
            if params.get("apply_border_filter", False):
                border_standoff = params.get("border_standoff", 15)
                mask = filter_by_border_endpoints(
                    mask,
                    border_standoff=border_standoff,
                    require_start_near_border=False,
                    require_end_near_border=False,
                    min_endpoint_near_border=1,
                )

            return mask

        # NEW: Topo-optimized hysteresis using feature-based threshold prediction
        # Matches "adaptive_topo_hysteresis" and "adaptive_topo_hysteresis_w_border"
        if method.startswith("adaptive_topo_hysteresis"):
            # Get topo-optimal thresholds based on probability features
            low_thresh, high_thresh = predict_topo_optimal_threshold(pr)

            # Override with params if provided
            if "low_thresh" in params:
                low_thresh = params["low_thresh"]
            if "high_thresh" in params:
                high_thresh = params["high_thresh"]

            min_size = params.get("min_size", 500)

            # Apply hysteresis thresholding
            mask = hysteresis_threshold(pr, low_threshold=low_thresh, high_threshold=high_thresh)
            mask = remove_small_components(mask, min_size=min_size)

            # Optional: Apply border filtering
            if params.get("apply_border_filter", False):
                border_standoff = params.get("border_standoff", 15)
                mask = filter_by_border_endpoints(
                    mask,
                    border_standoff=border_standoff,
                    require_start_near_border=False,
                    require_end_near_border=False,
                    min_endpoint_near_border=1,
                )

            return mask

        # Extract method variant if specified (e.g., "adaptive_learned", "adaptive_regression")
        if "_" in method:
            adaptive_method = method.split("_", 1)[1]
        else:
            adaptive_method = params.get("adaptive_method", "learned")

        min_size = params.get("min_size", 500)
        return adaptive_threshold_postprocess(pr, method=adaptive_method, min_size=min_size)

    # =========================================================================
    # GT-INFORMED POST-PROCESSING METHODS
    # Uses priors from analysis of 786 GT volumes (6129 ribbons)
    # =========================================================================

    # Handle GT-informed methods via the "method" parameter
    if "method" in params:
        gt_method = params["method"]
        # Create a copy of params without the "method" key
        gt_params = {k: v for k, v in params.items() if k != "method"}
        return postprocess_gt(pr, method=gt_method, params=gt_params)

    # Direct GT method calls (fallback for method names starting with "gt_")
    if method.startswith("gt_"):
        # Extract the GT method name (e.g., "gt_minimal_0.19" -> "gt_minimal")
        gt_method_parts = method.split("_")
        if len(gt_method_parts) >= 2:
            # Handle methods like "gt_minimal", "gt_balanced", "gt_topology", etc.
            gt_method = "_".join(gt_method_parts[:2])  # e.g., "gt_minimal"
            return postprocess_gt(pr, method=gt_method, params=params)

    # Handle hysteresis_gt methods
    if method.startswith("hysteresis_gt"):
        return hysteresis_gt_informed(
            pr,
            low_threshold=params.get("low_threshold", 0.12),
            high_threshold=params.get("high_threshold", 0.28),
        )

    # =========================================================================
    # PRECISION-FOCUSED POST-PROCESSING (2026-02-05)
    # Targets excessive false positives that tank topo & VOI on low-scoring samples
    # =========================================================================

    if method == "precision_adaptive":
        return precision_adaptive_postprocess(pr, **params)

    if method == "opening_cleanup":
        return opening_cleanup_postprocess(pr, **params)

    if method == "volume_calibrated":
        return volume_calibrated_postprocess(pr, **params)

    if method == "highconf_seed":
        return highconf_seed_postprocess(pr, **params)

    if method == "combined_precision":
        return combined_precision_postprocess(pr, **params)

    if method == "precision_border_aware":
        return precision_border_aware_postprocess(pr, **params)

    if method == "multiscale_consensus":
        return multiscale_consensus_postprocess(pr, **params)

    # =========================================================================
    # MICRO-HOLE FILLING POST-PROCESSING (2026-02-09)
    # Targets 1-2 voxel holes/pits inside ribbons that tank topo scores.
    # Topo metric inverts FG/BG, so micro-holes become spurious BG components.
    # =========================================================================

    if method == "holefill" or method == "adaptive_border_holefill":
        return adaptive_border_aware_holefill(pr, **params)

    if method == "holefill_v2" or method == "adaptive_border_holefill_v2":
        return adaptive_border_aware_holefill_v2(pr, **params)

    if method == "holefill_v3" or method == "adaptive_border_holefill_v3":
        return adaptive_border_aware_holefill_v3(pr, **params)

    raise ValueError(f"Unknown post-processing method: {method}")


def score_single_tif(
    gt_path,
    pred_paths_with_weights: list[tuple[Path, float]],
    postprocess_method: str = "threshold",
    postprocess_params: dict | None = None,
    filename: str | None = None,
    ensemble_method: str = "weighted_average",
    ensemble_params: dict | None = None,
    surface_tolerance=2.0,
    voi_connectivity=26,
    voi_transform="one_over_one_plus",
    voi_alpha=0.3,
    topo_weight=0.3,
    surface_dice_weight=0.35,
    voi_weight=0.35,
):
    """Score a single prediction against ground truth with post-processing.

    Args:
        gt_path: Path to ground truth TIFF
        pred_paths_with_weights: List of (prediction_path, weight) tuples
        postprocess_method: Post-processing method name
        postprocess_params: Parameters for post-processing
        filename: Sample filename (for cache lookup)
        ensemble_method: How to combine multiple models:
            - "weighted_average": Standard weighted mean
            - "geometric_mean": Emphasizes agreement
            - "harmonic_mean": Heavily penalizes disagreement
            - "maximum": Union-like (highest prob)
            - "minimum": Intersection-like (lowest prob)
            - "soft_or": Probabilistic union
            - "soft_and": Probabilistic intersection
            - "rank_average": Rank-based averaging
            - "confidence_weighted": Weight by prediction confidence
            - "adaptive": Boost agreements, penalize disagreements
            - "stacked_boost": Use confident predictions to boost uncertain
        ensemble_params: Extra parameters for ensemble method
        surface_tolerance: Surface dice tolerance
        voi_connectivity: VOI connectivity (6, 18, or 26)
        voi_transform: VOI transform type
        voi_alpha: VOI alpha parameter
        topo_weight: Weight for topology score
        surface_dice_weight: Weight for surface dice
        voi_weight: Weight for VOI score

    Returns:
        Score report object

    """
    gt: np.ndarray = load_volume(gt_path)

    # Combine predictions using specified ensemble method
    pr = combine_predictions(
        pred_paths_with_weights,
        load_npz_volume,
        ensemble_method=ensemble_method,
        ensemble_params=ensemble_params,
    )

    # Apply post-processing
    pr_binary = postprocess_prediction(pr, postprocess_method, postprocess_params, filename)

    score_report = compute_leaderboard_score(
        predictions=pr_binary,
        labels=gt,
        dims=(0, 1, 2),
        spacing=(1.0, 1.0, 1.0),
        surface_tolerance=surface_tolerance,
        voi_connectivity=voi_connectivity,
        voi_transform=voi_transform,
        voi_alpha=voi_alpha,
        combine_weights=(topo_weight, surface_dice_weight, voi_weight),
        fg_threshold=None,
        ignore_label=2,
        ignore_mask=None,
    )
    return score_report


# Global variable for ground truth directory (needed for worker processes)
GT_DIR = None


def _init_worker(gt_dir):
    """Initialize worker process with GT directory."""
    global GT_DIR
    GT_DIR = gt_dir


def score_pred_file(
    pred_info,
    method="threshold",
    params=None,
    ensemble_method="weighted_average",
    ensemble_params=None,
):
    """Worker function for parallel processing."""
    if params is None:
        params = {}
    if ensemble_params is None:
        ensemble_params = {}

    filename = Path(pred_info[0][0]).stem
    gt_path = os.path.join(GT_DIR, filename + ".tif")

    try:
        score_value = score_single_tif(
            gt_path,
            pred_info,
            postprocess_method=method,
            postprocess_params=params,
            filename=filename,
            ensemble_method=ensemble_method,
            ensemble_params=ensemble_params,
            surface_tolerance=2.0,
            voi_connectivity=26,
            voi_transform="one_over_one_plus",
            voi_alpha=0.3,
            topo_weight=0.3,
            surface_dice_weight=0.35,
            voi_weight=0.35,
        )
        return filename, score_value, None
    except Exception as e:
        import traceback

        return filename, None, f"{e!s}\n{traceback.format_exc()}"


def _is_gpu_method(method_name: str) -> bool:
    """Check if a method requires GPU processing."""
    return method_name.startswith("gpu_")


def _process_score_result(
    filename: str,
    score,
    error: str,
    method_name: str,
    model_config_name: str,
    ensemble_method: str,
    all_scores: list,
) -> None:
    """Process a single scoring result and append to all_scores."""
    if error:
        logging.error(f"ERROR scoring {filename}: {error}")
        return

    all_scores.append(
        {
            "filename": filename,
            "model_config": model_config_name,
            "ensemble_method": ensemble_method,
            "postprocess_method": method_name,
            "final": score.score,
            "topo": score.topo.toposcore,
            "voi": score.voi.voi_score,
            "surface_dice": score.surface_dice,
            "topo_dim_0": score.topo.topoF1_by_dim[0],
            "topo_dim_1": score.topo.topoF1_by_dim[1],
            "voi_merge": score.voi.voi_merge,
            "voi_split": score.voi.voi_split,
            "voi_total": score.voi.voi_total,
        },
    )
    logging.info(f"Scored {filename} with {method_name}: {all_scores[-1]['final']:.4f}")


def run_scoring_experiment(
    gt_dir: Path,
    paired_predictions: list,
    methods_config: dict,
    model_config_name: str = "default",
    num_workers: int | None = None,
) -> pd.DataFrame:
    """Run scoring for multiple post-processing methods.

    Args:
        gt_dir: Path to ground truth directory
        paired_predictions: List of prediction path pairs with weights
        methods_config: Dict of method_name -> params
            Each params dict can include:
            - "method": post-processing method name
            - "ensemble_method": how to combine models (default: "weighted_average")
            - "ensemble_params": extra params for ensemble method
            - Other params passed to post-processing
        model_config_name: Name of the model configuration (e.g., "model_L_only")
        num_workers: Number of parallel workers

    Returns:
        DataFrame with all scores

    Note:
        GPU methods (prefixed with 'gpu_') are precomputed in batch for better
        GPU utilization, then scored in parallel like CPU methods.

    """
    global GT_DIR
    GT_DIR = gt_dir

    if num_workers is None:
        num_workers = cpu_count()

    # Precompute all GPU Frangi results upfront for better GPU utilization
    precompute_gpu_frangi_batch(
        paired_predictions,
        methods_config,
        load_npz_volume,
    )

    all_scores = []

    for method_name, params in methods_config.items():
        # Make a copy to avoid modifying the original config
        params = dict(params) if params else {}

        # Extract ensemble settings from params
        ensemble_method = params.pop("ensemble_method", "weighted_average")
        ensemble_params = params.pop("ensemble_params", {})

        logging.info(f"\n{'=' * 60}")
        logging.info(f"Model Config: {model_config_name}")
        logging.info(f"Scoring with method: {method_name}")
        logging.info(f"Params: {params}")
        logging.info(f"Ensemble: {ensemble_method} {ensemble_params if ensemble_params else ''}")
        logging.info(f"{'=' * 60}")

        # All methods can now run in parallel (GPU results are precomputed)
        method_scores = []
        failed_predictions = []  # Track predictions that crash workers

        score_fn = partial(
            score_pred_file,
            method=method_name,
            params=params,
            ensemble_method=ensemble_method,
            ensemble_params=ensemble_params,
        )

        try:
            with ProcessPoolExecutor(
                max_workers=num_workers,
                initializer=_init_worker,
                initargs=(gt_dir,),
            ) as executor:
                # Submit all tasks individually so one crash doesn't kill everything
                future_to_pred = {
                    executor.submit(score_fn, pred_info): pred_info
                    for pred_info in paired_predictions
                }

                for future in as_completed(future_to_pred):
                    pred_info = future_to_pred[future]
                    try:
                        filename, score, error = future.result()
                        _process_score_result(
                            filename,
                            score,
                            error,
                            method_name,
                            model_config_name,
                            ensemble_method,
                            all_scores,
                        )
                        if score is not None:
                            method_scores.append(score.score)
                    except (BrokenExecutor, Exception) as exc:
                        pred_name = Path(pred_info[0][0]).stem
                        logging.exception(
                            f"Worker crashed for {pred_name} with {method_name}: {exc}",
                        )
                        failed_predictions.append(pred_info)

        except (BrokenExecutor, Exception) as exc:
            logging.exception(f"Process pool failed: {exc}. Collecting unfinished work.")
            # Any predictions not yet processed go into failed list
            processed_stems = {
                s["filename"] for s in all_scores if s["postprocess_method"] == method_name
            }
            for pred_info in paired_predictions:
                stem = Path(pred_info[0][0]).stem
                if stem not in processed_stems:
                    failed_predictions.append(pred_info)

        # Retry failed predictions sequentially in the main process
        if failed_predictions:
            logging.info(
                f"Retrying {len(failed_predictions)} failed samples sequentially (in-process)...",
            )
            _init_worker(gt_dir)  # Set GT_DIR in main process
            for pred_info in failed_predictions:
                try:
                    filename, score, error = score_fn(pred_info)
                    _process_score_result(
                        filename,
                        score,
                        error,
                        method_name,
                        model_config_name,
                        ensemble_method,
                        all_scores,
                    )
                    if score is not None:
                        method_scores.append(score.score)
                except Exception as exc:
                    pred_name = Path(pred_info[0][0]).stem
                    logging.exception(f"Sequential retry also failed for {pred_name}: {exc}")

        # Print average for this method
        if method_scores:
            avg_score = np.mean(method_scores)
            std_score = np.std(method_scores)
            logging.info(
                f"\n>>> [{model_config_name}] {method_name} AVERAGE: {avg_score:.4f} ± {std_score:.4f} (n={len(method_scores)})",
            )

    return pd.DataFrame(all_scores)


def setup_logging(log_dir: Path) -> Path:
    """Setup logging to both console and file."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"scoring_experiment_{timestamp}.log"

    # Create a custom formatter
    formatter = logging.Formatter("%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    # Setup root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    # Clear any existing handlers
    root_logger.handlers = []

    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    return log_file


def get_model_predictions_path(
    base_dir: Path,
    dataset_name: str,
    trainer: str = "nnUNetTrainer",
    plans: str = "nnUNetResEncUNetMPlans",
    config: str = "3d_fullres",
) -> Path:
    """Construct the predictions path for a given dataset.

    Args:
        base_dir: Base directory for results
        dataset_name: Dataset name (e.g., "Dataset500_VesuviusSurface")
        trainer: Trainer name (default: "nnUNetTrainer")
        plans: Plans name (default: "nnUNetResEncUNetMPlans")
        config: Configuration (default: "3d_fullres")

    Returns:
        Path to predictions directory

    """
    return base_dir / dataset_name / f"{trainer}__{plans}__{config}" / "predictions"


def load_model_predictions(base_dir: Path, dataset_name: str, **kwargs) -> list[Path]:
    """Load prediction files for a model.

    Args:
        base_dir: Base directory for results
        dataset_name: Dataset name
        **kwargs: Additional arguments for get_model_predictions_path (ignored, auto-discovered)

    Returns:
        Sorted list of prediction file paths

    """
    # Auto-discover the trainer/plans/config directory structure
    dataset_dir = base_dir / dataset_name
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

    # Find all subdirectories that might contain predictions
    # They typically follow pattern: <trainer>__<plans>__<config>
    pred_path = None
    for subdir in dataset_dir.iterdir():
        if subdir.is_dir():
            predictions_dir = subdir / "predictions"
            if predictions_dir.exists():
                pred_path = predictions_dir
                break

    if pred_path is None:
        raise FileNotFoundError(
            f"No predictions directory found under: {dataset_dir}\n"
            f"Expected to find a subdirectory with a 'predictions' folder inside.",
        )

    preds = sorted(pred_path.glob("*.npz"))
    if not preds:
        raise FileNotFoundError(f"No .npz files found in: {pred_path}")
    return preds


if __name__ == "__main__":
    # Use 'spawn' to avoid issues with fork and native libraries (cc3d, etc.)
    # This creates fresh Python interpreters for workers instead of forking
    multiprocessing.set_start_method("spawn", force=True)

    REPO_ROOT = Path(os.getenv("BASE_DIR", "."))  # repo root; override with the BASE_DIR env var

    # Ground Truth Path
    GT_DIR = REPO_ROOT / "scoring/inference_labels"

    # Base results path
    BASE_DIR = REPO_ROOT / "scoring/temp_results"

    # Setup logging
    log_file = setup_logging(BASE_DIR)

    logging.info("=" * 80)
    logging.info("COMPREHENSIVE MODEL ENSEMBLE & POST-PROCESSING COMPARISON")
    logging.info("=" * 80)
    logging.info(f"Log file: {log_file}")

    # =========================================================================
    # MODEL CONFIGURATION - Just specify dataset names!
    # =========================================================================

    # Individual models to evaluate (each will be scored separately)
    INDIVIDUAL_MODELS = [
        "Dataset320_VesuviusSurface",
    ]

    # Ensembles to evaluate: list of (name, [(dataset, weight), ...])
    # Each ensemble combines multiple models with specified weights
    ENSEMBLE_CONFIGS = [
        # Example: equal weight ensemble of all three
        # (
        #     "ensemble_500_320",
        #     [
        #         ("Dataset500_VesuviusSurface", 1.0),
        #         ("Dataset320_VesuviusSurface", 1.0),
        #     ],
        # ),
        # Example: weighted ensemble
        # ("ensemble_510_520_weighted", [
        #     ("Dataset510_VesuviusSurface", 0.3),
        #     ("Dataset520_VesuviusSurface", 0.7),
        # ]),
    ]

    # =========================================================================
    # POST-PROCESSING METHODS
    # Applied to all individual models and ensembles
    # =========================================================================
    POSTPROCESS_METHODS = {
        # Baseline: adaptive border-aware (proven best for most samples)
        "adaptive_border_aware": {
            "min_size": 500,
        },
        # MICRO-HOLE FILLING (2026-02-09): Fill 1-2 voxel holes in ribbons
        # Eliminates spurious BG components that hurt dim0 topo score.
        # 2D per-slice fill + 3D remove_small_holes, NO component filtering.
        # Tested on 8 samples: 2 wins, 0 losses, 6 ties.
        "holefill": {
            "min_size": 500,
            "fill_2d": True,
            "fill_3d_holes": True,
            "max_hole_size": 100,
        },
    }

    # =========================================================================
    # ENSEMBLE COMBINATION METHODS (only for ensembles)
    # =========================================================================
    ENSEMBLE_METHODS_TO_TEST = [
        ("weighted_average", {}),
        # ("geometric_mean", {}),
        # ("soft_or", {}),
    ]

    # =========================================================================
    # LOAD ALL MODEL PREDICTIONS
    # =========================================================================
    all_datasets = set(INDIVIDUAL_MODELS)
    for _, model_weights in ENSEMBLE_CONFIGS:
        for dataset, _ in model_weights:
            all_datasets.add(dataset)

    model_predictions: dict[str, list[Path]] = {}
    reference_filenames = None

    for dataset in sorted(all_datasets):
        try:
            preds = load_model_predictions(BASE_DIR, dataset)
            model_predictions[dataset] = preds
            logging.info(f"Loaded {len(preds)} predictions from {dataset}")

            # Verify all models have matching files
            filenames = [p.stem for p in preds]
            if reference_filenames is None:
                reference_filenames = filenames
            else:
                assert filenames == reference_filenames, (
                    f"Filename mismatch for {dataset}: expected {len(reference_filenames)} files"
                )
        except FileNotFoundError as e:
            logging.exception(f"Skipping {dataset}: {e}")

    if not model_predictions:
        logging.error("No model predictions found!")
        sys.exit(1)

    num_samples = len(next(iter(model_predictions.values())))

    # =========================================================================
    # CALCULATE TOTAL EXPERIMENTS
    # =========================================================================
    num_individual = len([d for d in INDIVIDUAL_MODELS if d in model_predictions])
    num_ensembles = len(ENSEMBLE_CONFIGS)

    total_experiments = (
        num_individual * len(POSTPROCESS_METHODS)  # Individual models
        + num_ensembles * len(POSTPROCESS_METHODS) * len(ENSEMBLE_METHODS_TO_TEST)  # Ensembles
    )

    logging.info(f"\nNumber of samples: {num_samples}")
    logging.info(f"Individual models: {num_individual}")
    logging.info(f"Ensemble configurations: {num_ensembles}")
    logging.info(f"Post-processing methods: {len(POSTPROCESS_METHODS)}")
    logging.info(f"Ensemble combination methods: {len(ENSEMBLE_METHODS_TO_TEST)}")
    logging.info(f"TOTAL EXPERIMENTS: {total_experiments}")
    logging.info(f"Estimated samples to score: {total_experiments * num_samples}")

    num_workers = max(1, cpu_count() // 2)  # Use half of available cores
    logging.info(f"Using {num_workers} CPU cores for parallel scoring...")

    # Collect all results
    all_results = []
    experiment_count = 0

    # =========================================================================
    # RUN INDIVIDUAL MODEL EXPERIMENTS
    # =========================================================================
    for dataset in INDIVIDUAL_MODELS:
        if dataset not in model_predictions:
            logging.warning(f"Skipping {dataset} - predictions not loaded")
            continue

        config_name = dataset.replace("Dataset", "D").replace("_VesuviusSurface", "")
        logging.info(f"\n{'#' * 80}")
        logging.info(f"INDIVIDUAL MODEL: {dataset}")
        logging.info(f"{'#' * 80}")

        # Create single-model predictions list
        preds = model_predictions[dataset]
        paired_predictions = [[(pred, 1.0)] for pred in preds]

        # Build methods config (no ensemble method needed for single model)
        methods_config = {}
        for pp_name, pp_params in POSTPROCESS_METHODS.items():
            methods_config[pp_name] = {
                **pp_params,
                "ensemble_method": "weighted_average",
                "ensemble_params": {},
            }

        # Run scoring
        config_results_df = run_scoring_experiment(
            GT_DIR,
            paired_predictions,
            methods_config,
            model_config_name=config_name,
            num_workers=num_workers,
        )

        all_results.append(config_results_df)
        experiment_count += len(methods_config)
        logging.info(f"Progress: {experiment_count}/{total_experiments} experiments complete")

    # =========================================================================
    # RUN ENSEMBLE EXPERIMENTS
    # =========================================================================
    for ensemble_name, model_weights in ENSEMBLE_CONFIGS:
        # Check all models are available
        missing = [d for d, _ in model_weights if d not in model_predictions]
        if missing:
            logging.warning(f"Skipping ensemble {ensemble_name} - missing: {missing}")
            continue

        logging.info(f"\n{'#' * 80}")
        logging.info(f"ENSEMBLE: {ensemble_name}")
        for dataset, weight in model_weights:
            logging.info(f"  - {dataset}: weight={weight}")
        logging.info(f"{'#' * 80}")

        # Create paired predictions with weights
        num_files = len(model_predictions[model_weights[0][0]])
        paired_predictions = []
        for i in range(num_files):
            pred_weight_list = [
                (model_predictions[dataset][i], weight) for dataset, weight in model_weights
            ]
            paired_predictions.append(pred_weight_list)

        # Test each ensemble method
        for ens_method_name, ens_params in ENSEMBLE_METHODS_TO_TEST:
            logging.info(f"\n{'=' * 60}")
            logging.info(f"Ensemble Method: {ens_method_name}")
            logging.info(f"{'=' * 60}")

            # Build methods config with current ensemble method
            methods_config = {}
            for pp_name, pp_params in POSTPROCESS_METHODS.items():
                methods_config[pp_name] = {
                    **pp_params,
                    "ensemble_method": ens_method_name,
                    "ensemble_params": ens_params,
                }

            # Run scoring
            config_results_df = run_scoring_experiment(
                GT_DIR,
                paired_predictions,
                methods_config,
                model_config_name=f"{ensemble_name}__{ens_method_name}",
                num_workers=num_workers,
            )

            all_results.append(config_results_df)
            experiment_count += len(methods_config)
            logging.info(f"Progress: {experiment_count}/{total_experiments} experiments complete")

    # Combine all results
    scores_df = pd.concat(all_results, ignore_index=True)

    # =========================================================================
    # RESULTS ANALYSIS
    # =========================================================================
    logging.info(f"\n{'=' * 80}")
    logging.info("COMPREHENSIVE RESULTS SUMMARY")
    logging.info(f"{'=' * 80}")
    logging.info(f"Total scored samples: {len(scores_df)}")

    avg_columns = [
        "final",
        "topo",
        "voi",
        "surface_dice",
        "topo_dim_0",
        "topo_dim_1",
        "voi_merge",
        "voi_split",
        "voi_total",
    ]

    # Group by model_config, ensemble_method, and postprocess_method
    summary = scores_df.groupby(
        ["model_config", "ensemble_method", "postprocess_method"],
    )[avg_columns].agg(["mean", "std"])

    # Flatten column names
    summary.columns = ["_".join(col).strip() for col in summary.columns.values]
    summary = summary.reset_index()

    # Sort by final score
    summary = summary.sort_values("final_mean", ascending=False)

    logging.info("\n" + "=" * 120)
    logging.info("TOP 30 CONFIGURATIONS BY FINAL SCORE")
    logging.info("=" * 120)
    logging.info(
        f"{'Model Config':<35} {'Ens Method':<20} {'PostProc':<25} {'Final':>12} {'Topo':>10} {'VOI':>10} {'Surf':>10}",
    )
    logging.info("-" * 120)

    for _, row in summary.head(30).iterrows():
        final = f"{row['final_mean']:.4f}±{row['final_std']:.4f}"
        topo = f"{row['topo_mean']:.4f}"
        voi = f"{row['voi_mean']:.4f}"
        surf = f"{row['surface_dice_mean']:.4f}"
        logging.info(
            f"{row['model_config']:<35} {row['ensemble_method']:<20} {row['postprocess_method']:<25} "
            f"{final:>12} {topo:>10} {voi:>10} {surf:>10}",
        )

    # Best configuration per metric
    logging.info(f"\n{'=' * 80}")
    logging.info("BEST CONFIGURATION PER METRIC")
    logging.info(f"{'=' * 80}")

    for metric in ["final", "topo", "voi", "surface_dice"]:
        best_idx = summary[f"{metric}_mean"].idxmax()
        best_row = summary.loc[best_idx]
        logging.info(
            f"{metric.upper():>15}: {best_row['model_config']} | {best_row['ensemble_method']} | "
            f"{best_row['postprocess_method']} ({best_row[f'{metric}_mean']:.4f})",
        )

    # Compare individual models vs ensembles
    logging.info(f"\n{'=' * 80}")
    logging.info("INDIVIDUAL MODELS vs ENSEMBLES (Best per category)")
    logging.info(f"{'=' * 80}")

    # Best individual models (configs that don't contain "__" are individual)
    individual_rows = summary[~summary["model_config"].str.contains("__")]
    if len(individual_rows) > 0:
        logging.info("Best Individual Models:")
        for model_config in individual_rows["model_config"].unique():
            model_rows = individual_rows[individual_rows["model_config"] == model_config]
            best = model_rows.loc[model_rows["final_mean"].idxmax()]
            logging.info(
                f"  {model_config:<20} {best['postprocess_method']:<25} Final={best['final_mean']:.4f}",
            )

    # Best ensembles (configs that contain "__" are ensembles)
    ensemble_rows = summary[summary["model_config"].str.contains("__")]
    if len(ensemble_rows) > 0:
        logging.info("\nBest Ensembles:")
        best_ens = ensemble_rows.loc[ensemble_rows["final_mean"].idxmax()]
        logging.info(
            f"  {best_ens['model_config']} | {best_ens['ensemble_method']} | "
            f"{best_ens['postprocess_method']} Final={best_ens['final_mean']:.4f}",
        )

    # Save detailed results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = BASE_DIR / f"comprehensive_results_{timestamp}.csv"
    scores_df.to_csv(output_path, index=False)
    logging.info(f"\nDetailed results saved to: {output_path}")

    # Save summary
    summary_path = BASE_DIR / f"comprehensive_summary_{timestamp}.csv"
    summary.to_csv(summary_path, index=False)
    logging.info(f"Summary saved to: {summary_path}")

    logging.info(f"\n{'=' * 80}")
    logging.info("EXPERIMENT COMPLETE")
    logging.info(f"{'=' * 80}")
    logging.info(f"Log file: {log_file}")
