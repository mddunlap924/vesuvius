#!/usr/bin/env python3
"""Test model ensembles and threshold optimization for scoring improvement."""

import csv
import gc
import os
import sys
from pathlib import Path

import numpy as np
from analyze_prediction_probabilities import adaptive_threshold_postprocess
from PIL import Image, ImageSequence
from post_processing_probs import filter_by_border_endpoints
from topometrics.leaderboard import compute_leaderboard_score

BASE_DIR = Path(os.getenv("BASE_DIR", "."))  # repo root; override with the BASE_DIR env var

GT_DIR = BASE_DIR / "scoring/inference_labels/"

MODEL_RESULTS_DIR = BASE_DIR / "scoring/temp_results"

MODEL_PATHS = {
    "D320": f"{MODEL_RESULTS_DIR}/Dataset320_VesuviusSurface/nnUNetTrainer_FineTune500epochs_LR001__nnUNetResEncUNetLPlans__3d_fullres/predictions/",
    "D500": f"{MODEL_RESULTS_DIR}/Dataset500_VesuviusSurface/nnUNetTrainer__nnUNetResEncUNetMPlans__3d_fullres/predictions/",
    "D510": f"{MODEL_RESULTS_DIR}/Dataset510_VesuviusSurface/nnUNetTrainer__nnUNetResEncUNetMPlans__3d_fullres/predictions/",
    "D520": f"{MODEL_RESULTS_DIR}/Dataset520_VesuviusSurface/nnUNetTrainer__nnUNetResEncUNetMPlans__3d_fullres/predictions/",
}

SAMPLES = sorted([p.stem for p in Path(MODEL_PATHS["D320"]).glob("*.npz")])


def load_gt(sample):
    gt_path = GT_DIR / f"{sample}.tif"
    img = Image.open(str(gt_path))
    frames = [np.array(f) for f in ImageSequence.Iterator(img)]
    return np.stack(frames, axis=0)


def load_probs(model, sample):
    path = Path(MODEL_PATHS[model]) / f"{sample}.npz"
    return np.load(str(path))["probabilities"][1].astype(np.float32)


def score_mask(mask, gt):
    r = compute_leaderboard_score(
        predictions=mask,
        labels=gt,
        dims=(0, 1, 2),
        spacing=(1.0, 1.0, 1.0),
        surface_tolerance=2.0,
        voi_connectivity=26,
        voi_transform="one_over_one_plus",
        voi_alpha=0.3,
        combine_weights=(0.3, 0.35, 0.35),
        fg_threshold=None,
        ignore_label=2,
        ignore_mask=None,
    )
    return r.score, r.topo.toposcore, r.voi.voi_score, r.surface_dice


def postprocess_adaptive(prob):
    """Standard adaptive threshold + border filter."""
    mask = adaptive_threshold_postprocess(prob, method="learned", min_size=500)
    mask = filter_by_border_endpoints(
        mask,
        border_standoff=15,
        require_start_near_border=False,
        require_end_near_border=False,
        min_endpoint_near_border=1,
    )
    return mask


def postprocess_fixed_threshold(prob, threshold=0.20, min_size=500):
    """Fixed threshold + border filter."""
    from scipy import ndimage
    from scipy.ndimage import label as scipy_label

    binary = (prob > threshold).astype(np.uint8)
    if min_size > 0:
        labeled, num = scipy_label(binary)
        if num > 0:
            sizes = ndimage.sum(binary, labeled, range(1, num + 1))
            for i, sz in enumerate(sizes, 1):
                if sz < min_size:
                    binary[labeled == i] = 0

    mask = filter_by_border_endpoints(
        binary,
        border_standoff=15,
        require_start_near_border=False,
        require_end_near_border=False,
        min_endpoint_near_border=1,
    )
    return mask


def run_experiment(name, get_prob_fn, postprocess_fn, samples=SAMPLES):
    """Run one experiment across all samples and return avg score."""
    scores = []
    details = []
    for i, sample in enumerate(samples):
        gt = load_gt(sample)
        prob = get_prob_fn(sample)
        mask = postprocess_fn(prob)
        s, topo, voi, sd = score_mask(mask, gt)
        scores.append(s)
        details.append((sample, s, topo, voi, sd))
        print(
            f"  [{name}] {i + 1}/30 {sample}: {s:.4f} (topo={topo:.4f} voi={voi:.4f} sd={sd:.4f})",
            flush=True,
        )
        del gt, prob, mask
        gc.collect()

    avg = np.mean(scores)
    print(f">>> {name}: avg={avg:.4f}", flush=True)
    return avg, details


def ensemble_prob_fn(models, weights):
    """Create a function that returns weighted ensemble of probabilities."""

    def fn(sample):
        result = None
        for m, w in zip(models, weights):
            p = load_probs(m, sample)
            if result is None:
                result = p * w
            else:
                result += p * w
            del p
        return result

    return fn


def main():
    results = {}

    # ===== PHASE 1: Ensembles with adaptive threshold =====
    print("=" * 60)
    print("PHASE 1: Ensemble averaging with adaptive threshold")
    print("=" * 60)

    ensemble_configs = [
        ("D320_only", ["D320"], [1.0]),
        ("D320+D500_eq", ["D320", "D500"], [0.5, 0.5]),
        ("D320+D500_70_30", ["D320", "D500"], [0.7, 0.3]),
        ("D320+D500_80_20", ["D320", "D500"], [0.8, 0.2]),
        ("D320+D520_eq", ["D320", "D520"], [0.5, 0.5]),
        ("D320+D500+D520_eq", ["D320", "D500", "D520"], [1 / 3, 1 / 3, 1 / 3]),
        ("D320+D500+D520_w", ["D320", "D500", "D520"], [0.5, 0.3, 0.2]),
        ("all4_eq", ["D320", "D500", "D510", "D520"], [0.25, 0.25, 0.25, 0.25]),
        ("all4_D320heavy", ["D320", "D500", "D510", "D520"], [0.5, 0.2, 0.1, 0.2]),
    ]

    for name, models, weights in ensemble_configs:
        avg, details = run_experiment(
            name,
            ensemble_prob_fn(models, weights),
            postprocess_adaptive,
        )
        results[name] = (avg, details)

    # ===== PHASE 2: Threshold optimization on best ensemble =====
    print("\n" + "=" * 60)
    print("PHASE 2: Threshold optimization on best ensemble")
    print("=" * 60)

    # Find best ensemble
    best_name = max(results, key=lambda k: results[k][0])
    best_avg = results[best_name][0]
    print(f"Best ensemble so far: {best_name} = {best_avg:.4f}")

    # Parse the best ensemble config
    best_config = None
    for name, models, weights in ensemble_configs:
        if name == best_name:
            best_config = (models, weights)
            break

    if best_config:
        models, weights = best_config
        # Test different fixed thresholds
        for threshold in [0.10, 0.12, 0.15, 0.18, 0.20, 0.22, 0.25]:
            tname = f"{best_name}_t{threshold:.2f}"
            avg, details = run_experiment(
                tname,
                ensemble_prob_fn(models, weights),
                lambda prob, t=threshold: postprocess_fixed_threshold(prob, threshold=t),
            )
            results[tname] = (avg, details)

    # ===== PHASE 3: Per-sample oracle =====
    print("\n" + "=" * 60)
    print("PHASE 3: Summary")
    print("=" * 60)

    # Print sorted results
    sorted_results = sorted(results.items(), key=lambda x: x[1][0], reverse=True)
    print(f"\n{'Method':<35} {'Avg Score':>10}")
    print("-" * 47)
    for name, (avg, _) in sorted_results:
        delta = avg - 0.6860
        marker = " *** BETTER" if avg > 0.6860 else ""
        print(f"{name:<35} {avg:>10.4f}  ({delta:+.4f}){marker}")

    # Per-sample oracle across ALL methods
    oracle_scores = []
    for i, sample in enumerate(SAMPLES):
        best_s = max(results[name][1][i][1] for name in results)
        oracle_scores.append(best_s)
    oracle_avg = np.mean(oracle_scores)
    print(f"\n{'Per-sample oracle':<35} {oracle_avg:>10.4f}  ({oracle_avg - 0.6860:+.4f})")

    # Save detailed results
    outpath = Path("/tmp/ensemble_results.csv")
    with open(outpath, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["method", "sample", "score", "topo", "voi", "surface_dice"])
        for name, (avg, details) in results.items():
            for sample, s, topo, voi, sd in details:
                writer.writerow(
                    [name, sample, f"{s:.6f}", f"{topo:.6f}", f"{voi:.6f}", f"{sd:.6f}"]
                )
    print(f"\nDetailed results saved to {outpath}")


if __name__ == "__main__":
    main()
