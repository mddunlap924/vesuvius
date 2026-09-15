import json
import os

# Pin the run to a single GPU. Override CUDA_VISIBLE_DEVICES to target a different device:
#   CUDA_VISIBLE_DEVICES=1 python scripts/nnunetbaseline/train_nnunetv2.py
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import shutil
import subprocess
from functools import partial
from multiprocessing import Pool
from pathlib import Path
from typing import List, Literal, Optional, Tuple, Union

# IMPORTANT: Set this BEFORE importing nnunetv2
# Blosc2 is a newer compression format but can cause compatibility issues
# Use blosc2 format (faster, smaller files)
# os.environ["NNUNET_ENABLE_AMP"] = "1"
os.environ["nnUNet_USE_BLOSC2"] = "1"
# os.environ["nnUNet_USE_BF16"] = "1"  # BF16 mixed precision on recent NVIDIA GPUs
# os.environ["NNUNET_DISABLE_TORCH_COMPILE"] = "1"
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
import tifffile
from tqdm.auto import tqdm

# =============================================================================
# TYPE DEFINITIONS
# =============================================================================

# Available epoch counts from nnUNet's built-in trainers
# These are pre-defined trainer classes in nnUNet - you cannot use arbitrary values
# See: nnunetv2/training/nnUNetTrainer/variants/training_length/nnUNetTrainer_Xepochs.py
Epochs = Literal[1, 5, 10, 20, 50, 100, 250, 500, 750, 1000, 2000, 4000, 8000]

# =============================================================================
# CONFIGURATION - MODIFY THESE FOR YOUR SETUP
# =============================================================================

BASE_DIR = Path(os.getenv("BASE_DIR", "."))  # repo root; override with the BASE_DIR env var

# Base Directory on local machine
BASE_DIR_LOCAL = BASE_DIR / "nnunet_training"

# Competition data path (Kaggle dataset)
INPUT_DIR = BASE_DIR / "data/raw_download"

# Pre-prepared nnUNet preprocessed dataset
# Upload your preprocessed data as a Kaggle dataset to skip the 1-2 hour preprocessing step
# Set to non-existent path to force fresh preprocessing
PREPARED_PREPROCESSED_PATH = BASE_DIR_LOCAL / "data/vesuvius-surface-nnunet-preprocessed"

# Working directories
# Large intermediate files (cleared between sessions)
WORKING_DIR = BASE_DIR_LOCAL / "temp"
OUTPUT_DIR = BASE_DIR_LOCAL / "working"  # Final outputs (persisted)

# nnUNet directory structure (follows nnUNet conventions)
NNUNET_BASE = WORKING_DIR / "nnUNet_data"
# Small - uses symlinks to competition data
NNUNET_RAW = NNUNET_BASE / "nnUNet_raw"
# Large - can use pre-prepared dataset
NNUNET_PREPROCESSED = NNUNET_BASE / "nnUNet_preprocessed"
# Trained models go here (persisted)
NNUNET_RESULTS = OUTPUT_DIR / "nnUNet_results"

# Dataset configuration
DATASET_ID = 520  # nnUNet dataset ID (arbitrary, just needs to be consistent)
DATASET_NAME = f"Dataset{DATASET_ID:03d}_VesuviusSurface"

# =============================================================================
# TRAINING CONFIGURATION
# =============================================================================

# Fold: 0-4 for 5-fold cross-validation, "all" for training on all data
# - Use "all" for final submission (faster, uses all training data)
# - Use 0-4 for model validation and ensembling
FOLD: int | str = "all"

# Configuration: determines network architecture and resolution
# - "2d": Fastest, processes slice-by-slice (good for quick experiments)
# - "3d_lowres": Fast, handles large volumes well (NOTE: creates predicted_next_stage for cascade)
# - "3d_fullres": Best quality, single-stage (RECOMMENDED)
# - "3d_cascade_fullres": Two-stage for very large volumes (slowest, uses 3d_lowres predictions)
CONFIGURATION = "3d_fullres"

# Planner: determines network architecture variant
# - "nnUNetPlanner": Standard U-Net encoder
# - "nnUNetPlannerResEncM": ResNet encoder, medium size (RECOMMENDED - often 1-2% better)
# - "nnUNetPlannerResEncL": ResNet encoder, large (more parameters, slower)
PLANNER = "nnUNetPlannerResEncM"

# Must match planner (see reference table in header)
PLANS_NAME = "nnUNetResEncUNetMPlans"

# Number of CPU workers for preprocessing and data preparation
NUM_WORKERS = os.cpu_count() or 4

# Epochs: Number of training epochs
# - None or 1000: Default full training
# - 50-100: Quick experiments
# - 250-500: Good balance (check progress.png to see if converged)
# Must be one of: 1, 5, 10, 20, 50, 100, 250, 500, 750, 1000, 2000, 4000, 8000
EPOCHS: Epochs | None = None

# Command timeout in seconds (None for no timeout)
# Useful for Kaggle's 9-hour limit - set to e.g., 28800 (8 hours) to leave buffer
COMMAND_TIMEOUT: int | None = None


def _get_gpu_count() -> int:
    """Get number of available CUDA GPUs."""
    try:
        import torch

        return torch.cuda.device_count() if torch.cuda.is_available() else 1
    except ImportError:
        return 0


# Number of GPUs for DDP training
# Auto-detected by default. Set to 1 to disable multi-GPU.
# NOTE: Multi-GPU DDP can sometimes hang in notebook environments.
# If training hangs, try num_gpus=1
# NUM_GPUS: int = _get_gpu_count()
NUM_GPUS = 1

# =============================================================================
# PATH HELPER FUNCTIONS
# =============================================================================


def _get_trainer_name_simple(epochs: int | None) -> str:
    """Get trainer class name based on epochs (simple version for path construction)."""
    if epochs is None or epochs == 1000:
        return "nnUNetTrainer"
    if epochs == 1:
        return "nnUNetTrainer_1epoch"  # Special case: singular form
    return f"nnUNetTrainer_{epochs}epochs"


def get_training_output_dir(
    epochs: Epochs | None = None,
    plans: str = PLANS_NAME,
    config: str = CONFIGURATION,
    fold: int | str = FOLD,
) -> Path:
    """Get the training output directory path based on configuration.

    nnUNet creates this folder structure:
    NNUNET_RESULTS/Dataset100_VesuviusSurface/nnUNetTrainer__nnUNetResEncUNetMPlans__3d_lowres/fold_all/

    Use this to find checkpoints, logs, and progress.png
    """
    _epochs = epochs if epochs is not None else EPOCHS
    trainer = _get_trainer_name_simple(_epochs)
    return NNUNET_RESULTS / DATASET_NAME / f"{trainer}__{plans}__{config}" / f"fold_{fold}"


def get_progress_image_path(
    epochs: Epochs | None = None,
    plans: str = PLANS_NAME,
    config: str = CONFIGURATION,
    fold: int | str = FOLD,
) -> Path:
    """Get path to training progress image (loss curves, metrics over epochs)."""
    return get_training_output_dir(epochs, plans, config, fold) / "progress.png"


# ----------------------------------------------------------------------------
# Environment Setup
# ----------------------------------------------------------------------------
def setup_environment():
    """Set up nnUNet environment variables and directories."""
    for d in [NNUNET_RAW, NNUNET_PREPROCESSED, NNUNET_RESULTS, OUTPUT_DIR]:
        d.mkdir(parents=True, exist_ok=True)

    os.environ["nnUNet_raw"] = str(NNUNET_RAW)
    os.environ["nnUNet_preprocessed"] = str(NNUNET_PREPROCESSED)
    os.environ["nnUNet_results"] = str(NNUNET_RESULTS)
    os.environ["nnUNet_compile"] = "true"

    print(f"nnUNet_raw: {NNUNET_RAW}")
    print(f"nnUNet_preprocessed: {NNUNET_PREPROCESSED}")
    print(f"nnUNet_results: {NNUNET_RESULTS}")
    print(f"nnUNet_USE_BLOSC2: {os.environ.get('nnUNet_USE_BLOSC2', 'not set')} (0=NPZ, 1=blosc2)")
    print(f"NUM_WORKERS: {NUM_WORKERS}")


def _link_prepared_preprocessed() -> bool:
    """Link/copy pre-prepared preprocessed data if available.

    Copies the folder structure and metadata files (which nnUNet may need to modify),
    but symlinks the heavy .npz/.b2nd data files to save space.

    Handles two possible structures:
    1. PREPARED_PREPROCESSED_PATH points directly to Dataset100_* folder
    2. PREPARED_PREPROCESSED_PATH contains Dataset100_* as subfolder

    Returns True if linked/copied successfully.
    """
    if not PREPARED_PREPROCESSED_PATH.exists():
        return False

    # Determine source directory - could be the path itself or a subfolder
    source_dir = PREPARED_PREPROCESSED_PATH
    exact_dataset_folder = source_dir / DATASET_NAME
    if exact_dataset_folder.exists():
        source_dir = exact_dataset_folder
    else:
        print(f"No dataset folder found in {PREPARED_PREPROCESSED_PATH}")
        return False

    # if not (source_dir / "dataset.json").exists():
    #     # Look for Dataset folder inside
    #     dataset_folders = list(PREPARED_PREPROCESSED_PATH.glob(
    #         f"Dataset*_{DATASET_NAME.split('_')[1]}*"))
    #     if not dataset_folders:
    #         dataset_folders = list(PREPARED_PREPROCESSED_PATH.glob("Dataset*"))
    #     if dataset_folders:
    #         source_dir = dataset_folders[0]
    #     else:
    #         print(f"No dataset folder found in {PREPARED_PREPROCESSED_PATH}")
    #         return False

    target_dir = NNUNET_PREPROCESSED / DATASET_NAME

    if target_dir.exists():
        print(f"Preprocessed data already exists: {target_dir}")
        return True

    print(f"Linking preprocessed data from: {source_dir}")
    target_dir.mkdir(parents=True, exist_ok=True)

    # Files that nnUNet may need to write/modify - copy these
    copy_patterns = ["*.json", "*.pkl", "*.txt"]

    # Heavy data files - symlink these
    symlink_patterns = ["*.npz", "*.npy", "*.b2nd"]

    copied = 0
    linked = 0

    for src_path in source_dir.rglob("*"):
        if src_path.is_dir():
            continue

        # Compute relative path and create target path
        rel_path = src_path.relative_to(source_dir)
        dst_path = target_dir / rel_path
        dst_path.parent.mkdir(parents=True, exist_ok=True)

        # Check if this is a heavy data file (symlink) or metadata (copy)
        is_data_file = any(src_path.match(pat) for pat in symlink_patterns)

        if is_data_file:
            if not dst_path.exists():
                dst_path.symlink_to(src_path.resolve())
                linked += 1
        elif not dst_path.exists():
            shutil.copy2(src_path, dst_path)
            copied += 1

    print(f"Prepared preprocessed data: {copied} files copied, {linked} files symlinked")
    print(f"Location: {target_dir}")
    return True


setup_environment()

# ----------------------------------------------------------------------------
# Installation & Imports
# ----------------------------------------------------------------------------
# Show GPU configuration
print(f"Available GPUs: {_get_gpu_count()}")
print(f"Using NUM_GPUS={NUM_GPUS}")

# ----------------------------------------------------------------------------
# Data Utilities
# ----------------------------------------------------------------------------


def create_spacing_json(output_path: Path, shape: tuple, spacing: tuple = (1.0, 1.0, 1.0)):
    """Create JSON sidecar with spacing info for TIFF files."""
    json_data = {"spacing": list(spacing)}
    with open(output_path, "w") as f:
        json.dump(json_data, f)


def load_nifti(path: Path) -> np.ndarray:
    """Load NIfTI file (used for loading nnUNet predictions)."""
    return nib.load(str(path)).get_fdata()


def create_dataset_json(output_dir: Path, num_training: int, file_ending: str = ".tif") -> dict:
    """Create dataset.json with ignore label support and 3D TIFF reader."""
    dataset_json = {
        "channel_names": {"0": "CT"},
        "labels": {"background": 0, "surface": 1, "ignore": 2},
        "numTraining": num_training,
        "file_ending": file_ending,
        "overwrite_image_reader_writer": "SimpleTiffIO",
    }

    json_path = output_dir / "dataset.json"
    with open(json_path, "w") as f:
        json.dump(dataset_json, f, indent=4)

    print(f"Created {json_path}")
    print(f"  - {num_training} training cases")
    print("  - Labels: background(0), surface(1), ignore(2)")
    print("  - Reader: SimpleTiffIO (3D TIFF)")

    return dataset_json


# ----------------------------------------------------------------------------
# Dataset Preparation
# ----------------------------------------------------------------------------


def prepare_single_case(
    src_path: Path,
    dest_path: Path,
    json_path: Path,
    use_symlinks: bool = True,
) -> bool:
    """Prepare a single TIFF file for nnUNet: create symlink/copy and JSON sidecar.
    Returns True on success, False on failure.
    """
    try:
        # Get shape for JSON
        with tifffile.TiffFile(src_path) as tif:
            shape = (
                tif.pages[0].shape if len(tif.pages) == 1 else (len(tif.pages), *tif.pages[0].shape)
            )

        # Link or copy file
        if use_symlinks:
            if not dest_path.exists():
                dest_path.symlink_to(src_path.resolve())
        else:
            shutil.copy2(src_path, dest_path)

        # Create JSON sidecar
        create_spacing_json(json_path, shape)
        return True

    except Exception as e:
        print(f"Error processing {src_path.name}: {e}")
        return False


def _prepare_training_case(
    img_path: Path,
    train_labels_dir: Path,
    images_dir: Path,
    labels_dir: Path,
    use_symlinks: bool,
) -> bool:
    """Worker function for parallel dataset preparation."""
    case_id = img_path.stem
    label_path = train_labels_dir / img_path.name

    if not label_path.exists():
        return False

    img_ok = prepare_single_case(
        img_path,
        images_dir / f"{case_id}_0000.tif",
        images_dir / f"{case_id}_0000.json",
        use_symlinks,
    )

    label_ok = prepare_single_case(
        label_path,
        labels_dir / f"{case_id}.tif",
        labels_dir / f"{case_id}.json",
        use_symlinks,
    )

    return img_ok and label_ok


def prepare_dataset(input_dir: Path, max_cases: int | None = None, use_symlinks: bool = True):
    """Convert competition data to nnUNet format using TIFF directly (no NIfTI).
    Uses multiprocessing for faster preparation.

    Competition structure:
    - train_images/*.tif  (3D volumes)
    - train_labels/*.tif  (3D labels: 0=bg, 1=surface, 2=ignore)
    """
    dataset_dir = NNUNET_RAW / DATASET_NAME
    images_dir = dataset_dir / "imagesTr"
    labels_dir = dataset_dir / "labelsTr"

    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    train_images_dir = input_dir / "train_images"
    train_labels_dir = input_dir / "train_labels"

    if not train_images_dir.exists():
        print(f"ERROR: {train_images_dir} not found!")
        return None

    image_files = sorted(train_images_dir.glob("*.tif"))
    if max_cases:
        image_files = image_files[:max_cases]

    print(f"Found {len(image_files)} training cases")
    print(f"Using {'symlinks' if use_symlinks else 'copy'}")
    print(f"Processing with {NUM_WORKERS} workers...")

    # Create worker function with fixed arguments
    worker = partial(
        _prepare_training_case,
        train_labels_dir=train_labels_dir,
        images_dir=images_dir,
        labels_dir=labels_dir,
        use_symlinks=use_symlinks,
    )

    # Process in parallel with progress bar
    with Pool(NUM_WORKERS) as pool:
        results = list(
            tqdm(
                pool.imap(worker, image_files),
                total=len(image_files),
                desc="Preparing dataset",
            ),
        )

    num_converted = sum(results)
    create_dataset_json(dataset_dir, num_converted, file_ending=".tif")

    print(f"\nDataset prepared: {num_converted} cases")
    print(f"Location: {dataset_dir}")

    return dataset_dir


# ----------------------------------------------------------------------------
# Synthetic Data (for testing)
# ----------------------------------------------------------------------------


def create_single_synthetic_case(
    case_id: str,
    size: tuple[int, int, int],
    images_dir: Path,
    labels_dir: Path,
):
    """Create a single synthetic training case."""
    # Create volume with structure
    volume = np.random.randn(*size).astype(np.float32) * 0.1

    # Add spherical shell as "surface"
    z, y, x = np.ogrid[: size[0], : size[1], : size[2]]
    center = np.array(size) // 2
    dist = np.sqrt((z - center[0]) ** 2 + (y - center[1]) ** 2 + (x - center[2]) ** 2)
    shell = (dist > 15) & (dist < 20)
    volume[shell] += 1.0

    # Create labels
    labels = np.zeros(size, dtype=np.uint8)
    labels[shell] = 1  # Surface

    # Add ignore regions (label 2)
    ignore_mask = np.random.random(size) < 0.15
    labels[ignore_mask] = 2

    # Save as TIFF
    tifffile.imwrite(images_dir / f"{case_id}_0000.tif", volume)
    tifffile.imwrite(labels_dir / f"{case_id}.tif", labels)

    # Create JSON sidecars
    create_spacing_json(images_dir / f"{case_id}_0000.json", size)
    create_spacing_json(labels_dir / f"{case_id}.json", size)


def create_synthetic_dataset(num_cases: int = 5, size: tuple[int, int, int] = (64, 64, 64)):
    """Create synthetic 3D data with ignore regions for testing (TIFF format)."""
    dataset_dir = NNUNET_RAW / DATASET_NAME
    images_dir = dataset_dir / "imagesTr"
    labels_dir = dataset_dir / "labelsTr"

    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    for i in tqdm(range(num_cases), desc="Creating synthetic data"):
        case_id = f"case_{i:03d}"
        create_single_synthetic_case(case_id, size, images_dir, labels_dir)

    create_dataset_json(dataset_dir, num_cases, file_ending=".tif")
    print(f"Created synthetic dataset: {dataset_dir}")

    return dataset_dir


# To use synthetic data instead of real data, uncomment:
# create_synthetic_dataset(num_cases=5)

# ----------------------------------------------------------------------------
# nnUNet Commands
# ----------------------------------------------------------------------------


def _run_command(
    cmd: str,
    name: str = "Command",
    tail_lines: int = 20,
    timeout: int | None = COMMAND_TIMEOUT,
) -> bool:
    """Execute shell command and handle output parsing.

    Args:
        cmd: Shell command to execute
        name: Display name for logging
        tail_lines: Number of stdout lines to show on success
        timeout: Timeout in seconds (None for no timeout)

    Returns:
        True if command succeeded, False otherwise

    """
    print(f"Running: {cmd}")
    if timeout:
        print(f"Timeout: {timeout}s ({timeout / 3600:.1f}h)")

    try:
        result = subprocess.run(
            cmd,
            check=False,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        print(f"{name} TIMEOUT after {timeout}s!")
        return False

    if result.returncode != 0:
        print(f"{name} FAILED!")
        print(f"STDERR:\n{result.stderr[-3000:]}")
        return False

    print(f"{name} complete!")
    if result.stdout.strip():
        lines = result.stdout.strip().split("\n")
        print("\n".join(lines[-tail_lines:]))

    return True


def run_preprocessing(
    dataset_id: int = DATASET_ID,
    planner: str = PLANNER,
    num_workers: int = NUM_WORKERS,
    configurations: list[str] | None = None,
    timeout: int | None = COMMAND_TIMEOUT,
) -> bool:
    """Run nnUNet preprocessing.

    Args:
        dataset_id: nnUNet dataset ID
        planner: Planner class name
        num_workers: Number of CPU workers for parallel processing
        configurations: List of configs to preprocess (e.g., ["3d_fullres"])
        timeout: Timeout in seconds (None for no timeout)

    Returns:
        True if preprocessing succeeded

    """
    if configurations is None:
        configurations = [CONFIGURATION]

    cmd = f"nnUNetv2_plan_and_preprocess -d {dataset_id:03d} -np {num_workers}"
    cmd += f" -pl {planner}"
    cmd += f" -c {' '.join(configurations)}"
    # cmd += f" -gpu_memory_target 23"

    return _run_command(cmd, "Preprocessing", timeout=timeout)


def _get_trainer_name(epochs: Epochs | None) -> str:
    """Get trainer class name based on epochs."""
    if epochs is None or epochs == 1000:
        return "nnUNetTrainer"
    if epochs == 1:
        return "nnUNetTrainer_1epoch"
    return f"nnUNetTrainer_{epochs}epochs"


def run_training(
    dataset_id: int = DATASET_ID,
    config: str = CONFIGURATION,
    fold: int | str = FOLD,
    plans: str = PLANS_NAME,
    epochs: Epochs | None = EPOCHS,
    pretrained_weights: Path | None = None,
    continue_training: bool = False,
    only_run_validation: bool = False,
    disable_checkpointing: bool = False,
    npz: bool = False,
    num_gpus: int = NUM_GPUS,
    timeout: int | None = COMMAND_TIMEOUT,
) -> bool:
    """Run nnUNet training.

    Args:
        dataset_id: nnUNet dataset ID
        config: Configuration name (3d_fullres, 2d, etc.)
        fold: Fold number (0-4) or "all" for training on all data
        plans: Plans name matching the planner used
        epochs: Number of epochs. Available: 1, 5, 10, 20, 50, 100, 150, 200, 250,
                300, 400, 500, 750, 1000, 2000, 4000, 8000. None = 1000 (default)
        pretrained_weights: Optional path to pretrained checkpoint for fine-tuning
        continue_training: Continue from last checkpoint (add -c flag)
        only_run_validation: Only run validation, skip training
        disable_checkpointing: Disable saving checkpoints (saves disk space)
        npz: Save softmax outputs during validation (needed for ensembling)
        num_gpus: Number of GPUs for DDP training (default: auto-detected)
        timeout: Timeout in seconds (None for no timeout)

    Returns:
        True if training succeeded

    Note:
        For multi-GPU (DDP) training, batch size should be divisible by num_gpus.
        The first run extracts preprocessed data - wait for GPU usage before
        starting additional folds on other GPUs.

        When using fold="all", nnUNet will run validation on ALL training data
        after training completes, which can be slow. This is normal behavior.

    """
    trainer = _get_trainer_name(epochs)
    cmd = f"nnUNetv2_train {dataset_id:03d} {config} {fold} -p {plans} -tr {trainer}"

    if pretrained_weights:
        cmd += f" -pretrained_weights {pretrained_weights}"
    if continue_training:
        # Find checkpoint to resume from: prefer checkpoint_final.pth, fallback to checkpoint_best.pth
        model_dir = get_training_output_dir(epochs=epochs, plans=plans, config=config, fold=fold)
        checkpoint_final = model_dir / "checkpoint_final.pth"
        checkpoint_best = model_dir / "checkpoint_best.pth"
        if checkpoint_final.exists():
            print(f"Resuming from: {checkpoint_final}")
            cmd += " --c"
        elif checkpoint_best.exists():
            print(f"Resuming from: {checkpoint_best}")
            cmd += " --c"
        else:
            print(f"WARNING: No checkpoint found in {model_dir}, starting fresh")
    if only_run_validation:
        cmd += " --val"
    if disable_checkpointing:
        cmd += " --disable_checkpointing"
    if npz:
        cmd += " --npz"
    if num_gpus > 1:
        cmd += f" -num_gpus {num_gpus}"

    epochs_str = epochs if epochs else 1000
    gpu_str = f", {num_gpus} GPUs" if num_gpus > 1 else ""
    return _run_command(
        cmd,
        f"Training ({epochs_str} epochs{gpu_str})",
        tail_lines=30,
        timeout=timeout,
    )


def run_inference(
    input_dir: Path,
    output_dir: Path,
    dataset_id: int = DATASET_ID,
    config: str = CONFIGURATION,
    fold: int | str = FOLD,
    plans: str = PLANS_NAME,
    epochs: Epochs | None = EPOCHS,
    save_probabilities: bool = True,
    num_processes_preprocessing: int = 2,
    num_processes_segmentation: int = 2,
    timeout: int | None = COMMAND_TIMEOUT,
) -> bool:
    """Run inference with trained model.

    Args:
        input_dir: Directory with test images (must have _0000 suffix)
        output_dir: Directory to save predictions
        dataset_id: nnUNet dataset ID
        config: Configuration name
        fold: Fold number used for training (or "all" or tuple like "0,1,2")
        plans: Plans name
        epochs: Epochs used during training (must match trained model)
        save_probabilities: Whether to save probability maps (.npz files)
        num_processes_preprocessing: Parallel processes for preprocessing
        num_processes_segmentation: Parallel processes for segmentation
        timeout: Timeout in seconds (None for no timeout)

    Returns:
        True if inference succeeded

    """
    output_dir.mkdir(parents=True, exist_ok=True)

    trainer = _get_trainer_name(epochs)

    cmd = f"nnUNetv2_predict -d {dataset_id:03d} -c {config} -f {fold}"
    cmd += f" -i {input_dir} -o {output_dir} -p {plans} -tr {trainer}"
    cmd += f" -npp {num_processes_preprocessing} -nps {num_processes_segmentation}"
    cmd += " --verbose"

    if save_probabilities:
        cmd += " --save_probabilities"

    return _run_command(cmd, "Inference", timeout=timeout)


# ----------------------------------------------------------------------------
# Test Data Preparation
# ----------------------------------------------------------------------------


def prepare_test_data(input_dir: Path, output_dir: Path, use_symlinks: bool = True) -> Path:
    """Prepare test TIFF images for nnUNet inference."""
    output_dir.mkdir(parents=True, exist_ok=True)

    test_images_dir = input_dir / "test_images"

    if not test_images_dir.exists():
        print(f"ERROR: {test_images_dir} not found!")
        return output_dir

    test_files = sorted(test_images_dir.glob("*.tif"))
    print(f"Found {len(test_files)} test cases")
    print(f"Using {'symlinks' if use_symlinks else 'copy'}")

    for img_path in tqdm(test_files, desc="Preparing test data"):
        case_id = img_path.stem
        prepare_single_case(
            img_path,
            output_dir / f"{case_id}_0000.tif",
            output_dir / f"{case_id}_0000.json",
            use_symlinks,
        )

    return output_dir


# ----------------------------------------------------------------------------
# Inference Utilities
# ----------------------------------------------------------------------------


def load_probabilities(npz_path: Path) -> np.ndarray:
    """Load probability maps from nnUNet inference.

    Only available if inference was run with save_probabilities=True.
    Shape: (num_classes, D, H, W) with float32 values in [0, 1].
    """
    data = np.load(npz_path)
    return data["probabilities"]


def predictions_to_tiff(pred_dir: Path, output_dir: Path):
    """Convert nnUNet predictions to 3D TIFF files.

    nnUNet outputs:
    - .npz files with probability maps (if save_probabilities=True)
    - .tif files with predictions (our SimpleTiffIO format)
    - .pkl files with metadata

    This function:
    1. First tries to load .npz files and convert probabilities to binary predictions
    2. Falls back to .tif files if .npz not found
    3. Saves as uint8 TIFF (0=background, 1=surface)
    """
    output_dir.mkdir(parents=True, exist_ok=True)

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
            tifffile.imwrite(output_dir / f"{case_id}.tif", pred)
    elif tif_files:
        print(f"Copying {len(tif_files)} TIFF prediction files...")
        for tif_path in tqdm(tif_files, desc="Copying TIFF"):
            case_id = tif_path.stem
            # Load and ensure uint8
            pred = tifffile.imread(str(tif_path)).astype(np.uint8)
            tifffile.imwrite(output_dir / f"{case_id}.tif", pred)
    # Try NIfTI as last resort (legacy format)
    elif nii_files:
        print(f"Converting {len(nii_files)} NIfTI files to TIFF...")
        for nii_path in tqdm(nii_files, desc="Converting to TIFF"):
            case_id = nii_path.stem.replace(".nii", "")
            pred = load_nifti(nii_path).astype(np.uint8)
            tifffile.imwrite(output_dir / f"{case_id}.tif", pred)
    else:
        print(f"WARNING: No prediction files found in {pred_dir}")
        print("  Checked for: *.npz, *.tif, *.nii.gz")


# ----------------------------------------------------------------------------
# Full Pipeline
# ----------------------------------------------------------------------------


def _parse_model_path(
    model_path: str | Path | None,
) -> tuple[Path | None, int | None, str | None, str | None, str | None]:
    """Parse model path to extract configuration parameters.

    Model path format: .../DatasetXXX_Name/TrainerName__PlansName__Config/fold_X/checkpoint.pth

    Returns:
        (model_dir, epochs, plans, config, fold) - extracted from path, None if not found

    Note: Parses path structure even if file doesn't exist (for path validation).

    """
    if model_path is None:
        return None, None, None, None, None

    model_path = Path(model_path)

    # Determine model directory from path (even if doesn't exist)
    # If path ends with .pth, use parent directory
    if model_path.suffix == ".pth" or (model_path.exists() and model_path.is_file()):
        model_dir = model_path.parent
    else:
        model_dir = model_path

    # Try to parse folder structure
    # Expected: .../fold_X or .../TrainerName__PlansName__Config/fold_X
    try:
        parts = model_dir.parts

        # Find fold
        fold = None
        for part in reversed(parts):
            if part.startswith("fold_"):
                fold = part.replace("fold_", "")
                break

        # Find trainer__plans__config
        epochs = None
        plans = None
        config = None
        for part in parts:
            if "__" not in part:
                continue
            segments = part.split("__")
            if len(segments) >= 3:
                trainer_name = segments[0]
                plans = segments[1]
                config = segments[2]
                # Extract epochs from trainer name
                if "epochs" in trainer_name:
                    import re

                    match = re.search(r"(\d+)epochs?", trainer_name)
                    if match:
                        epochs = int(match.group(1))
                elif trainer_name == "nnUNetTrainer":
                    epochs = 1000  # Default
                break

        return model_dir, epochs, plans, config, fold
    except Exception:
        return model_dir, None, None, None, None


def full_pipeline(
    # Data options
    max_cases: int | None = None,
    # Stage control
    do_preprocess: bool = True,
    do_train: bool = True,
    do_inference: bool = True,
    # Training options (all tunable parameters)
    config: str = CONFIGURATION,
    fold: int | str = FOLD,
    planner: str = PLANNER,
    plans: str = PLANS_NAME,
    epochs: Epochs | None = EPOCHS,
    pretrained_weights: Path | None = None,
    continue_training: bool = False,
    num_gpus: int = NUM_GPUS,
    # Inference options
    save_probabilities: bool = True,
    # External model (for inference without training)
    model_path: str | Path | None = None,
    # Timeout
    timeout: int | None = COMMAND_TIMEOUT,
):
    """Run complete pipeline: setup -> data prep -> preprocess -> train -> predict.

    Args:
        max_cases: Limit number of training cases (None = use all)
        do_preprocess: Run preprocessing step
        do_train: Run training step
        do_inference: Run inference step

        config: nnUNet configuration (3d_fullres, 2d, 3d_lowres, 3d_cascade_fullres)
        fold: Fold number (0-4) or "all" for training on all data
        planner: Planner class name
        plans: Plans name matching the planner
        epochs: Number of training epochs (1, 5, 10, 20, 50, 100, 150, 200, 250,
                300, 400, 500, 750, 1000, 2000, 4000, 8000). None = 1000
        pretrained_weights: Path to pretrained weights for fine-tuning
        continue_training: Continue from last checkpoint
        num_gpus: Number of GPUs for DDP training

        save_probabilities: Save probability maps during inference

        model_path: Path to trained model checkpoint or directory (str or Path).
                    When provided with do_train=False, parameters (epochs, plans, config, fold)
                    are auto-extracted from the path if possible.
                    Example: "/path/to/nnUNetTrainer_5epochs__nnUNetResEncUNetMPlans__3d_fullres/fold_all/checkpoint_best.pth"
        timeout: Command timeout in seconds (None for no timeout)

    Returns:
        True if pipeline completed successfully

    """
    print("=" * 60)
    print("Vesuvius Surface Detection - nnUNet Pipeline")
    print("=" * 60)
    print(f"Stages: preprocess={do_preprocess}, train={do_train}, inference={do_inference}")
    print(f"Config: {config}, Fold: {fold}, Epochs: {epochs or 1000}, GPUs: {num_gpus}")

    # 1. Setup
    print("\n[1/5] Environment setup...")
    setup_environment()

    # 2. Prepare raw data (always - uses symlinks, fast)
    print("\n[2/5] Preparing raw dataset (symlinks)...")
    raw_dataset_dir = NNUNET_RAW / DATASET_NAME
    if not raw_dataset_dir.exists():
        prepare_dataset(INPUT_DIR, max_cases=max_cases)
    else:
        print(f"Raw dataset already exists: {raw_dataset_dir}")

    # 3. Preprocessing
    if do_preprocess:
        # Check if pre-prepared preprocessed data exists
        if _link_prepared_preprocessed():
            print("\n[3/5] Using pre-prepared preprocessed data...")
        else:
            print("\n[3/5] Preprocessing...")
            success = run_preprocessing(planner=planner, configurations=[config], timeout=timeout)
            if not success:
                print("Preprocessing failed!")
                return False
    else:
        print("\n[3/5] Skipping preprocessing...")
        _link_prepared_preprocessed()  # Still link if available

    # 4. Training
    if do_train:
        print("\n[4/5] Training...")
        success = run_training(
            config=config,
            fold=fold,
            plans=plans,
            epochs=epochs,
            pretrained_weights=pretrained_weights,
            continue_training=continue_training,
            num_gpus=num_gpus,
            timeout=timeout,
        )
        if not success:
            print("Training failed!")
            return False
    else:
        print("\n[4/5] Skipping training...")

        # Parse model_path to extract configuration if provided
        if model_path is not None:
            model_path = Path(model_path) if isinstance(model_path, str) else model_path
            _, parsed_epochs, parsed_plans, parsed_config, parsed_fold = _parse_model_path(
                model_path,
            )

            # Override parameters with parsed values
            if parsed_epochs is not None:
                epochs = parsed_epochs
                print(f"  Detected epochs={epochs} from model path")
            if parsed_plans is not None:
                plans = parsed_plans
                print(f"  Detected plans={plans} from model path")
            if parsed_config is not None:
                config = parsed_config
                print(f"  Detected config={config} from model path")
            if parsed_fold is not None:
                fold = parsed_fold
                print(f"  Detected fold={fold} from model path")

            if model_path.exists():
                print(f"Using model: {model_path}")
            else:
                print(f"WARNING: model_path does not exist: {model_path}")

        # Verify model exists
        expected_model_dir = get_training_output_dir(
            epochs=epochs,
            plans=plans,
            config=config,
            fold=fold,
        )
        expected_model_dir.mkdir(exist_ok=True, parents=True)
        checkpoint_final = expected_model_dir / "checkpoint_final.pth"
        checkpoint_best = expected_model_dir / "checkpoint_best.pth"

        if model_path and Path(model_path).exists():
            # Symlink external -> final so nnUNet finds it
            checkpoint_final.symlink_to(model_path)
            print(f"Map model: {checkpoint_final}")
        elif checkpoint_final.exists():
            print(f"Found model: {checkpoint_final}")
        elif checkpoint_best.exists():
            # Symlink best -> final so nnUNet finds it
            checkpoint_final.symlink_to(checkpoint_best)
            print(f"Found model: {checkpoint_best} (symlinked to checkpoint_final.pth)")
        elif do_inference:
            print(f"WARNING: No model found at {expected_model_dir}")
            print("  Provide model_path to a valid nnUNet checkpoint file")

    # 5. Inference
    if do_inference:
        print("\n[5/5] Running inference...")

        # Prepare test data in temp location
        test_input_dir = WORKING_DIR / "test_input"
        prepare_test_data(INPUT_DIR, test_input_dir)

        # Run inference
        predictions_dir = WORKING_DIR / "predictions"
        success = run_inference(
            test_input_dir,
            predictions_dir,
            config=config,
            fold=fold,
            plans=plans,
            epochs=epochs,
            save_probabilities=save_probabilities,
            timeout=timeout,
        )
        if not success:
            print("Inference failed!")
            return False

        # Convert predictions to TIFF
        print("\nConverting predictions to TIFF...")
        tiff_output_dir = OUTPUT_DIR / "predictions_tiff"
        predictions_to_tiff(predictions_dir, tiff_output_dir)

        print(f"\nPredictions saved to: {tiff_output_dir}")
    else:
        print("\n[5/5] Skipping inference...")

    print("\n" + "=" * 60)
    print("Pipeline complete!")
    print("=" * 60)

    # Show training progress if training was done
    if do_train:
        print("\n[Visualization] Training progress:")
        show_progress(epochs=epochs, plans=plans, config=config, fold=fold)

    # Visualize predictions if inference was done
    if do_inference:
        print("\n[Visualization] Sample prediction:")
        visualize_predictions(num_samples=1)

    return True


# ----------------------------------------------------------------------------
# Create Submission
# ----------------------------------------------------------------------------
def generate_submission(
    predictions_tiff_dir: Path = OUTPUT_DIR / "predictions_tiff",
    output_zip: Path = OUTPUT_DIR / "submission.zip",
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


# Example usage:
# generate_submission()
# generate_submission(delete_after_zip=True)  # Delete TIFFs after zipping to save space


# ----------------------------------------------------------------------------
# Visualization
# ----------------------------------------------------------------------------
def show_progress(
    epochs: Epochs | None = None,
    plans: str = PLANS_NAME,
    config: str = CONFIGURATION,
    fold: int | str = FOLD,
):
    """Display training progress image from nnUNet.

    Args:
        epochs: Number of epochs used for training (to find correct folder)
        plans: Plans name
        config: Configuration name
        fold: Fold number or "all"

    """
    progress_path = get_progress_image_path(epochs, plans, config, fold)

    if not progress_path.exists():
        print(f"Progress image not found: {progress_path}")
        print("Training may not have started or completed yet.")
        return

    from IPython.display import Image, display

    print(f"Training progress: {progress_path}")
    display(Image(filename=str(progress_path)))


def plot_three_axis_cuts(
    image_vol_path: Path,
    mask_vol_path: Path,
    figsize: tuple = (12, 15),
):
    """Plot middle slices of XY, XZ, and YZ planes for image volume and predicted mask.

    Args:
        image_vol_path: Path to input image TIFF
        mask_vol_path: Path to prediction mask TIFF
        figsize: Figure size (width, height)

    """
    print(f"Visualizing: {image_vol_path.name}")

    # Load volumes
    image_vol = tifffile.imread(str(image_vol_path))
    mask_vol = tifffile.imread(str(mask_vol_path)).astype(np.uint8)

    # Get dimensions
    d, h, w = image_vol.shape
    z_mid, y_mid, x_mid = d // 2, h // 2, w // 2

    # Extract slices
    slices = {
        "XY Plane (Z-axis)": (image_vol[z_mid, :, :], mask_vol[z_mid, :, :]),
        "XZ Plane (Y-axis)": (image_vol[:, y_mid, :], mask_vol[:, y_mid, :]),
        "YZ Plane (X-axis)": (image_vol[:, :, x_mid], mask_vol[:, :, x_mid]),
    }

    fig, axes = plt.subplots(3, 2, figsize=figsize)
    for i, (plane_name, (img_slice, mask_slice)) in enumerate(slices.items()):
        # Image Volume
        axes[i, 0].imshow(img_slice, cmap="gray")
        axes[i, 0].set_title(f"{plane_name} - Image Volume")
        axes[i, 0].axis("off")

        # Mask
        axes[i, 1].imshow(mask_slice, cmap="gray")
        axes[i, 1].set_title(f"{plane_name} - Predicted Mask")
        axes[i, 1].axis("off")

    plt.tight_layout()
    plt.show()


def visualize_predictions(
    predictions_dir: Path = OUTPUT_DIR / "predictions_tiff",
    test_images_dir: Path = INPUT_DIR / "test_images",
    num_samples: int = 1,
):
    """Visualize prediction results by showing image/mask pairs.

    Args:
        predictions_dir: Directory containing prediction TIFFs
        test_images_dir: Directory containing input test images
        num_samples: Number of samples to visualize

    """
    if not predictions_dir.exists():
        print(f"Predictions directory not found: {predictions_dir}")
        return

    predictions = sorted(predictions_dir.glob("*.tif"))
    if not predictions:
        print(f"No TIFF predictions found in {predictions_dir}")
        return

    print(f"Found {len(predictions)} predictions")

    for pred_path in predictions[:num_samples]:
        image_path = test_images_dir / pred_path.name
        if image_path.exists():
            plot_three_axis_cuts(image_path, pred_path)
        else:
            print(f"Warning: Input image not found: {image_path}")


# ----------------------------------------------------------------------------
# Run Pipeline
# ----------------------------------------------------------------------------
# Edit the block below to select the stage you want to run.
#
#   Full pipeline with defaults:
#       full_pipeline()
#   Quick experiment:
#       full_pipeline(do_preprocess=False, epochs=1000, config="3d_fullres", fold="all")
#   Inference only (a trained model already exists):
#       full_pipeline(do_preprocess=False, do_train=False)
#   Generate the submission ZIP:
#       generate_submission()

if __name__ == "__main__":
    full_pipeline(
        do_preprocess=False,
        epochs=1000,
        config="3d_fullres",
        fold="all",
    )
    generate_submission()
