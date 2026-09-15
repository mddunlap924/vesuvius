import json
import os
import shutil
import subprocess
import sys
import threading
from functools import partial
from multiprocessing import Pool
from pathlib import Path
from typing import List, Literal, Optional, Tuple, Union

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
import tifffile
from tqdm.auto import tqdm

# IMPORTANT: Set this BEFORE importing nnunetv2
# Blosc2 is a newer compression format but can cause compatibility issues
# Use blosc2 format (faster, smaller files)
os.environ["nnUNet_USE_BLOSC2"] = "1"

COMMAND_TIMEOUT: int | None = None


def _get_gpu_count() -> int:
    """Get number of available CUDA GPUs without initializing CUDA context."""
    try:
        # Use nvidia-smi to count GPUs without initializing PyTorch CUDA
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=count", "--format=csv,noheader"],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            # nvidia-smi returns one line per GPU, count the lines
            return len([l for l in result.stdout.strip().split("\n") if l])
    except Exception:
        pass
    # Fallback: try torch but this may initialize CUDA
    try:
        import torch

        return torch.cuda.device_count() if torch.cuda.is_available() else 0
    except ImportError:
        return 0


# Number of GPUs for DDP training
# Auto-detected by default. Set to 1 to disable multi-GPU.
# NOTE: Multi-GPU DDP can sometimes hang in notebook environments.
# If training hangs, try num_gpus=1
NUM_GPUS: int = _get_gpu_count()
print(f"Available GPUs: {_get_gpu_count()}")


def create_spacing_json(output_path: Path, shape: tuple, spacing: tuple = (1.0, 1.0, 1.0)):
    """Create JSON sidecar with spacing info for TIFF files."""
    json_data = {"spacing": list(spacing)}
    with open(output_path, "w") as f:
        json.dump(json_data, f)


def load_probabilities(npz_path: Path) -> np.ndarray:
    """Load probability maps from nnUNet inference.

    Only available if inference was run with save_probabilities=True.
    Shape: (num_classes, D, H, W) with float32 values in [0, 1].
    """
    data = np.load(npz_path)
    return data["probabilities"]


def predictions_to_tiff(pred_dir: Path, output_dir: Path, compute_checksums: bool = False):
    """Convert nnUNet predictions to 3D TIFF files.

    nnUNet outputs:
    - .npz files with probability maps (if save_probabilities=True)
    - .tif files with predictions (our SimpleTiffIO format)
    - .pkl files with metadata

    This function:
    1. First tries to load .npz files and convert probabilities to binary predictions
    2. Falls back to .tif files if .npz not found
    3. Saves as uint8 TIFF (0=background, 1=surface)

    Args:
        pred_dir: Directory containing prediction files
        output_dir: Directory to save TIFF files
        compute_checksums: If True, compute and print checksums/statistics for verification

    """
    import hashlib

    output_dir.mkdir(parents=True, exist_ok=True)

    verification_results = []

    # Try NPZ files first (probability maps)
    npz_files = list(pred_dir.glob("*.npz"))
    tif_files = list(pred_dir.glob("*.tif"))
    nii_files = list(pred_dir.glob("*.nii.gz"))

    if npz_files:
        print(f"Converting {len(npz_files)} NPZ probability files to TIFF...")
        for npz_path in tqdm(npz_files, desc="Converting to TIFF"):
            case_id = npz_path.stem
            # Load probabilities and take argmax to get class predictions
            probs = load_probabilities(npz_path)
            pred = np.argmax(probs, axis=0).astype(np.uint8)
            output_path = output_dir / f"{case_id}.tif"
            tifffile.imwrite(output_path, pred)

            if compute_checksums:
                # Compute MD5 checksum
                with open(output_path, "rb") as f:
                    file_hash = hashlib.md5(f.read()).hexdigest()
                # Compute statistics
                verification_results.append(
                    {
                        "filename": f"{case_id}.tif",
                        "md5": file_hash,
                        "shape": pred.shape,
                        "mean": float(pred.mean()),
                        "std": float(pred.std()),
                        "min": int(pred.min()),
                        "max": int(pred.max()),
                        "unique_values": sorted([int(v) for v in np.unique(pred)]),
                    },
                )
    elif tif_files:
        print(f"Copying {len(tif_files)} TIFF prediction files...")
        for tif_path in tqdm(tif_files, desc="Copying TIFF"):
            case_id = tif_path.stem
            # Load and ensure uint8
            pred = tifffile.imread(str(tif_path)).astype(np.uint8)
            output_path = output_dir / f"{case_id}.tif"
            tifffile.imwrite(output_path, pred)

            if compute_checksums:
                # Compute MD5 checksum
                with open(output_path, "rb") as f:
                    file_hash = hashlib.md5(f.read()).hexdigest()
                # Compute statistics
                verification_results.append(
                    {
                        "filename": f"{case_id}.tif",
                        "md5": file_hash,
                        "shape": pred.shape,
                        "mean": float(pred.mean()),
                        "std": float(pred.std()),
                        "min": int(pred.min()),
                        "max": int(pred.max()),
                        "unique_values": sorted([int(v) for v in np.unique(pred)]),
                    },
                )
    else:
        print(f"WARNING: No prediction files found in {pred_dir}")
        print("  Checked for: *.npz, *.tif, *.nii.gz")

    # Print verification results
    if compute_checksums and verification_results:
        print("\n" + "=" * 80)
        print("VERIFICATION CHECKSUMS (for comparing results across machines)")
        print("=" * 80)

        # Sort by filename for consistent ordering
        verification_results.sort(key=lambda x: x["filename"])

        # Print first 5 files in detail
        print("\nFirst 5 files (detailed):")
        for i, result in enumerate(verification_results[:5], 1):
            print(f"\n{i}. {result['filename']}")
            print(f"   MD5:           {result['md5']}")
            print(f"   Shape:         {result['shape']}")
            print(f"   Mean:          {result['mean']:.6f}")
            print(f"   Std:           {result['std']:.6f}")
            print(f"   Min/Max:       {result['min']} / {result['max']}")
            print(f"   Unique values: {result['unique_values']}")

        # Print summary table for all files
        print(f"\nAll {len(verification_results)} files (summary):")
        print("-" * 80)
        print(f"{'Filename':<30} {'MD5':<34} {'Mean':<12} {'Std':<10}")
        print("-" * 80)
        for result in verification_results:
            print(
                f"{result['filename']:<30} {result['md5']:<34} {result['mean']:<12.6f} {result['std']:<10.6f}",
            )
        print("=" * 80)

        # Save to JSON file for programmatic comparison
        import json

        verification_file = output_dir / "verification_checksums.json"
        with open(verification_file, "w") as f:
            json.dump(verification_results, f, indent=2)
        print(f"\nVerification data saved to: {verification_file}")
        print("You can use this file to programmatically compare results across machines.")
        print("=" * 80)


# ----------------------------------------------------------------------------
# Create Submission
# ----------------------------------------------------------------------------
def generate_submission(
    predictions_tiff_dir: Path,
    output_zip: Path,
    delete_after_zip: bool = True,  # Default True to save space on Kaggle
) -> Path | None:
    """Create submission ZIP from TIFF predictions.

    Args:
        predictions_tiff_dir: Directory containing predicted TIFF files
        output_zip: Output ZIP file path
        delete_after_zip: Delete TIFF files after adding to ZIP (saves space)

    Returns:
        Path to submission ZIP if successful, None otherwise

    """
    import zipfile

    if not predictions_tiff_dir.exists():
        print(f"ERROR: Predictions directory not found: {predictions_tiff_dir}")
        print("Run inference first!")
        return None

    tiff_files = sorted(predictions_tiff_dir.glob("*.tif"))

    if not tiff_files:
        print(f"No TIFF files found in {predictions_tiff_dir}")
        return None

    print(f"Creating submission ZIP with {len(tiff_files)} files...")

    with zipfile.ZipFile(output_zip, "w", zipfile.ZIP_DEFLATED) as zipf:
        for tiff_path in tqdm(tiff_files, desc="Zipping predictions"):
            # Add file with just the filename (no directory structure)
            zipf.write(tiff_path, tiff_path.name)

            if delete_after_zip:
                tiff_path.unlink()

    zip_size_mb = output_zip.stat().st_size / (1024 * 1024)
    print(f"Submission saved: {output_zip} ({zip_size_mb:.1f} MB)")

    return output_zip


def inference_only_multi_gpu(
    input_tiff_dir: str | Path,
    model_base_dir: str | Path,
    dataset_name: str,
    output_dir: str | Path = "/kaggle/working",
    num_gpus: int = 2,
    num_processes_preprocessing: int = 2,
    num_processes_segmentation: int = 2,
    save_probabilities: bool = True,
    generate_zip: bool = True,
    delete_tiffs_after_zip: bool = True,
    checkpoint_name: str = "checkpoint_best.pth",
    step_size: float = 0.5,
    disable_tta: bool = False,
    compute_verification_checksums: bool = False,
    trainer_override: str | None = None,
) -> dict | None:
    """Run inference using multiple GPUs in parallel.

    Uses nnUNet's recommended approach with --part_id and --num_parts flags
    to split the dataset across GPUs. Each GPU runs nnUNetv2_predict on a
    different subset of the data, all writing to the same output directory.

    Args:
        input_tiff_dir: Directory containing test TIFF images (*.tif files)
        model_base_dir: Base directory containing:
            - fold_*/checkpoint_best.pth or checkpoint_final.pth
            - dataset.json
            - plans.json
        output_dir: Output directory for results (default: /kaggle/working)
        num_gpus: Number of GPUs to use (default: 2)
        num_processes_preprocessing: Parallel processes for preprocessing per GPU
        num_processes_segmentation: Parallel processes for segmentation per GPU
        save_probabilities: Save probability maps (.npz files)
        generate_zip: Generate submission.zip file
        delete_tiffs_after_zip: Delete TIFF files after zipping to save space
        checkpoint_name: Checkpoint file to use (default: checkpoint_best.pth)
        step_size: Sliding window step size (0.5 = 50% overlap). Larger = faster
            but less accurate. Range: 0.0-1.0. Default: 0.5
        disable_tta: Disable test-time augmentation (mirroring). Faster but
        compute_verification_checksums: Compute and print MD5 checksums and statistics
            for verification across machines. Default: False
            slightly less accurate. Default: False (TTA enabled)
        trainer_override: Override the trainer class name. Use "nnUNetTrainer" to run
            inference with models trained using custom trainers that aren't installed.
            The base nnUNetTrainer works for inference since only weights are needed.

    Returns:
        Dictionary containing:
        - 'predictions_dir': Path to predictions directory
        - 'submission_zip': Path to submission ZIP (if generated)

    """
    import torch

    print("=" * 60)
    print("nnUNet Multi-GPU Inference Pipeline")
    print("=" * 60)

    # Convert paths
    input_tiff_dir = Path(input_tiff_dir)
    model_base_dir = Path(model_base_dir)
    output_dir = Path(output_dir)

    # Detect available GPUs
    available_gpu_count = torch.cuda.device_count()
    num_gpus = min(num_gpus, available_gpu_count)
    print(f"\n  Available GPUs: {available_gpu_count}")
    print(f"  Using GPUs: {num_gpus}")

    if num_gpus < 1:
        raise RuntimeError("No CUDA GPUs available")

    # Find fold directories
    fold_dirs = sorted(model_base_dir.glob("fold_*"))
    if not fold_dirs:
        raise FileNotFoundError(f"No fold_* directories found in: {model_base_dir}")

    # Detect which folds have checkpoints
    available_folds = []
    for fold_dir in fold_dirs:
        checkpoint_path = fold_dir / checkpoint_name
        if not checkpoint_path.exists():
            alt_checkpoint = fold_dir / (
                "checkpoint_final.pth" if "best" in checkpoint_name else "checkpoint_best.pth"
            )
            if alt_checkpoint.exists():
                checkpoint_path = alt_checkpoint
        if checkpoint_path.exists():
            fold_name = fold_dir.name.replace("fold_", "")
            try:
                available_folds.append(int(fold_name))
            except ValueError:
                available_folds.append(fold_name)

    if not available_folds:
        raise FileNotFoundError(
            f"No checkpoints found in any fold_* directory under: {model_base_dir}",
        )

    # Validate inputs
    if not input_tiff_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_tiff_dir}")
    if not model_base_dir.exists():
        raise FileNotFoundError(f"Model base directory not found: {model_base_dir}")
    if not (model_base_dir / "dataset.json").exists():
        raise FileNotFoundError(f"dataset.json not found in: {model_base_dir}")
    if not (model_base_dir / "plans.json").exists():
        raise FileNotFoundError(f"plans.json not found in: {model_base_dir}")

    print("\n[1/5] Model configuration")
    print(f"  Model base: {model_base_dir}")
    print(f"  Available folds: {available_folds}")
    print(f"  Checkpoint: {checkpoint_name}")

    # Set up output directories
    print("\n[2/5] Setting up output directories...")
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_dir = output_dir / "predictions"
    predictions_dir.mkdir(parents=True, exist_ok=True)
    print(f"  Output: {output_dir}")
    print(f"  Predictions: {predictions_dir}")

    # Prepare test input directory with JSON sidecars
    print("\n[3/5] Preparing test data (symlinks + JSON sidecars)...")
    test_input_dir = output_dir / "test_input"
    test_input_dir.mkdir(parents=True, exist_ok=True)

    # Handle both cases: input_dir contains test_images/ subfolder or is the folder with TIFFs
    if (input_tiff_dir / "test_images").exists():
        test_images_source = input_tiff_dir / "test_images"
    elif list(input_tiff_dir.glob("*.tif")):
        test_images_source = input_tiff_dir
    else:
        raise FileNotFoundError(f"No test TIFF images found in {input_tiff_dir}")

    test_files = sorted(test_images_source.glob("*.tif"))
    print(f"  Found {len(test_files)} test cases in {test_images_source}")

    for img_path in tqdm(test_files, desc="Creating symlinks + JSON sidecars"):
        case_id = img_path.stem
        dest_tiff = test_input_dir / f"{case_id}_0000.tif"
        dest_json = test_input_dir / f"{case_id}_0000.json"

        # Create symlink to original TIFF (no data duplication)
        if not dest_tiff.exists():
            dest_tiff.symlink_to(img_path.resolve())

        # Create JSON sidecar with spacing info
        if not dest_json.exists():
            with tifffile.TiffFile(img_path) as tif:
                shape = (
                    tif.pages[0].shape
                    if len(tif.pages) == 1
                    else (len(tif.pages), *tif.pages[0].shape)
                )
            create_spacing_json(dest_json, shape)

    # Extract model configuration from path
    # Path format: .../TrainerName__PlansName__Config/fold_X
    model_parent = model_base_dir
    try:
        parts = model_parent.name.split("__")
        if len(parts) == 3:
            trainer_name, plans_name, config_name = parts
        else:
            # Fallback to defaults
            trainer_name = "nnUNetTrainer"
            plans_name = "nnUNetPlans"
            config_name = "3d_fullres"
    except Exception:
        trainer_name = "nnUNetTrainer"
        plans_name = "nnUNetPlans"
        config_name = "3d_fullres"

    # Apply trainer override if provided (useful for custom trainers not installed locally)
    original_trainer_name = trainer_name
    if trainer_override:
        print(f"  Overriding trainer: {trainer_name} -> {trainer_override}")
        trainer_name = trainer_override

    # Extract dataset ID from dataset.json
    try:
        with open(model_base_dir / "dataset.json") as f:
            dataset_json = json.load(f)
        # Try to get dataset name from the json, or use a default
        dataset_name = dataset_json.get("name", dataset_name)
        # Extract ID from name if it follows DatasetXXX_Name format
        if dataset_name.startswith("Dataset"):
            dataset_id = int(dataset_name[7:10])
        else:
            raise ValueError("Invalid dataset name format")
    except Exception:
        raise ValueError("Failed to extract dataset ID from dataset.json")

    # Set up nnUNet results directory with proper structure
    # nnUNet expects: {nnUNet_results}/DatasetXXX_Name/Trainer__Plans__Config/
    # IMPORTANT: We must use the ORIGINAL trainer name for the directory structure
    # because nnUNet parses the folder name to determine the trainer class.
    # The -tr flag can override behavior but not the class loading.
    nnunet_results_dir = output_dir / "nnUNet_results"
    nnunet_results_dir.mkdir(parents=True, exist_ok=True)
    dataset_dir = nnunet_results_dir / dataset_name
    dataset_dir.mkdir(parents=True, exist_ok=True)

    # Use ORIGINAL trainer name (before override) for the symlink path
    # This ensures nnUNet can find and load the correct trainer class
    original_trainer_config_name = f"{original_trainer_name}__{plans_name}__{config_name}"
    model_link_in_results = dataset_dir / original_trainer_config_name

    # Create symlink: nnUNet_results/DatasetXXX/Trainer__Plans__Config -> actual model directory
    # Using original trainer name so nnUNet can find the checkpoint files
    if model_link_in_results.exists() or model_link_in_results.is_symlink():
        # Remove existing symlink/directory
        if model_link_in_results.is_symlink():
            model_link_in_results.unlink()
        elif model_link_in_results.is_dir():
            # Check if it already points to the right place
            if model_link_in_results.resolve() == model_base_dir.resolve():
                print(f"  Using existing model link: {model_link_in_results}")
            else:
                raise FileExistsError(
                    f"Model directory exists but is not a symlink: {model_link_in_results}",
                )
        else:
            model_link_in_results.unlink()

    # Create fresh symlink if needed
    if not model_link_in_results.exists():
        model_link_in_results.symlink_to(model_base_dir.resolve())
        print(f"  Created symlink: {model_link_in_results} -> {model_base_dir}")

    # Validate that nnUNet can find all required files through the symlink
    required_files = ["dataset.json", "plans.json"]
    for fold in available_folds:
        fold_dir = "fold_all" if fold == "all" else f"fold_{fold}"
        required_files.append(f"{fold_dir}/{checkpoint_name}")

    missing_files = []
    for file_path in required_files:
        full_path = model_link_in_results / file_path
        if not full_path.exists():
            missing_files.append(str(full_path))

    if missing_files:
        raise FileNotFoundError(
            "Required files not accessible through symlink. Missing:\n"
            + "\n".join(f"  - {f}" for f in missing_files),
        )

    # Build folds string for nnUNetv2_predict
    folds_str = ",".join(str(f) for f in available_folds)

    print(f"\n[4/5] Running parallel inference on {num_gpus} GPUs...")
    print("  Using nnUNet's --part_id/--num_parts for data splitting")
    print(f"  Dataset: {dataset_id}, Config: {config_name}")
    print(f"  Trainer: {trainer_name}, Plans: {plans_name}")
    print(f"  Folds: {folds_str}")
    print(
        f"  Step size: {step_size}, TTA: {'disabled' if disable_tta else 'enabled (mirroring on all axes)'}",
    )

    # Launch nnUNetv2_predict for each GPU with --part_id and --num_parts
    # This follows the official nnUNet multi-GPU inference instructions exactly
    processes = {}

    for gpu_id in range(num_gpus):
        # Build nnUNetv2_predict command - simplified to match official instructions
        # Official format: CUDA_VISIBLE_DEVICES=X nnUNetv2_predict -d ID -f fold -c config -i input -o output -num_parts N -part_id X
        cmd = [
            "nnUNetv2_predict",
            "-d",
            str(dataset_id),
            "-f",
            folds_str,
            "-c",
            config_name,
            "-i",
            str(test_input_dir),
            "-o",
            str(predictions_dir),
            "-num_parts",
            str(num_gpus),
            "-part_id",
            str(gpu_id),
        ]

        # Only add optional arguments if they differ from defaults
        if plans_name != "nnUNetPlans":
            cmd.extend(["-p", plans_name])
        # If trainer was overridden, we must pass it explicitly via -tr
        # because nnUNet will try to infer it from the directory name
        if trainer_override:
            cmd.extend(["-tr", trainer_name])
        elif trainer_name != "nnUNetTrainer":
            # Also add if trainer differs from default (no override case)
            cmd.extend(["-tr", trainer_name])
        if checkpoint_name != "checkpoint_final.pth":
            cmd.extend(["-chk", checkpoint_name])
        if step_size != 0.5:
            cmd.extend(["-step_size", str(step_size)])
        if disable_tta:
            cmd.append("--disable_tta")
        if save_probabilities:
            cmd.append("--save_probabilities")

        # Set up minimal environment - only CUDA_VISIBLE_DEVICES and nnUNet_results
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        env["nnUNet_results"] = str(nnunet_results_dir)
        # Force memory-efficient behavior for newer PyTorch versions
        env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

        print(f"  Launching GPU {gpu_id} (part {gpu_id}/{num_gpus})...")
        print(f"    CUDA_VISIBLE_DEVICES={gpu_id}")
        print(f"    Command: {' '.join(cmd)}")

        # Launch subprocess
        process = subprocess.Popen(
            cmd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        processes[gpu_id] = process

        # Add small delay between launches to avoid initialization race conditions
        if gpu_id < num_gpus - 1:
            import time

            time.sleep(2)

    # Helper function to stream output from subprocess
    def stream_output(process, gpu_id):
        """Stream output from subprocess and prefix with GPU ID."""
        for line in process.stdout:
            print(f"[GPU {gpu_id}] {line}", end="")

    # Start threads to stream output from each process
    output_threads = {}
    for gpu_id, proc in processes.items():
        thread = threading.Thread(target=stream_output, args=(proc, gpu_id))
        thread.daemon = True
        thread.start()
        output_threads[gpu_id] = thread

    # Wait for all processes to complete
    failed_gpus = []
    for gpu_id, proc in processes.items():
        return_code = proc.wait()
        output_threads[gpu_id].join(timeout=5)

        if return_code == 0:
            print(f"  GPU {gpu_id} finished successfully")
        else:
            print(f"  GPU {gpu_id} ERROR: Process exited with code {return_code}")
            failed_gpus.append(gpu_id)

    if failed_gpus:
        raise RuntimeError(f"GPU workers failed: {failed_gpus}")

    print(f"  All {num_gpus} GPU workers completed!")

    # Collect results
    results = {
        "predictions_dir": predictions_dir,
        "submission_zip": None,
    }

    # Convert to TIFF
    print("\n[5/5] Converting predictions to TIFF...")
    tiff_output_dir = output_dir / "predictions_tiff"
    predictions_to_tiff(
        predictions_dir,
        tiff_output_dir,
        compute_checksums=compute_verification_checksums,
    )

    # Generate submission ZIP
    if generate_zip:
        print("\nGenerating submission ZIP...")
        zip_path = generate_submission(
            predictions_tiff_dir=tiff_output_dir,
            output_zip=output_dir / "submission.zip",
            delete_after_zip=delete_tiffs_after_zip,
        )
        results["submission_zip"] = zip_path

    print("\n" + "=" * 60)
    print("Multi-GPU Inference complete!")
    print("=" * 60)
    print(f"GPUs used: {num_gpus}")
    print(f"Cases processed: {len(test_files)}")
    print(f"Predictions: {predictions_dir}")
    print(f"TIFF outputs: {tiff_output_dir}")
    if results["submission_zip"]:
        print(f"Submission: {results['submission_zip']}")

    return results
