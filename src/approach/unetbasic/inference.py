"""Inference script for BasicUNet - Load trained model and generate predictions.

This script mirrors the training setup to ensure consistent preprocessing and
uses sliding window inference for memory-efficient predictions on large volumes.

Usage:
    Single GPU:
        python src/approach/unetbasic/inference.py --experiment exp_v0 --checkpoint outputs/monai_swinunetr/checkpoint-XXXX

    Multi-GPU:
        CUDA_VISIBLE_DEVICES=0 python src/approach/unetbasic/inference.py --experiment exp_v0 --checkpoint outputs/monai_swinunetr/checkpoint-XXXX
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import tifffile
import torch
from monai import transforms as MT
from monai.inferers import sliding_window_inference
from monai.transforms import (
    Compose,
    EnsureChannelFirstd,
    LoadImaged,
    Rand3DElasticd,
    RandAdjustContrastd,
    RandCoarseDropoutd,
    RandCropByPosNegLabeld,
    RandFlipd,
    RandGaussianNoised,
    RandGaussianSmoothd,
    RandRotate90d,
    RandScaleIntensityd,
    RandShiftIntensityd,
    RandZoomd,
    ScaleIntensityd,
    ScaleIntensityRanged,
    SpatialCropd,
    SpatialPadd,
)
from omegaconf import DictConfig, OmegaConf
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from approach.unetbasic.models.unet import TopologyPreservingUNet

warnings.filterwarnings("ignore", category=UserWarning)

# Default BASE_DIR (project root) - can be overridden via --base_dir argument
DEFAULT_BASE_DIR: Path = Path(__file__).resolve().parent.parent.parent.parent
BASE_DIR: Path = DEFAULT_BASE_DIR


class PapyrusDataset(Dataset):
    def __init__(
        self,
        data_dir: Path,
        image_ids: list[str],
        patch_size=(96, 320, 320),
        augmentation: bool = True,
        augmentation_config: Any | None = None,
    ):
        self.data_dir = data_dir
        self.image_ids = image_ids
        self.augmentation = augmentation
        self.patch_size = patch_size

        self.transforms = Compose(
            [
                # LoadImaged(keys=["image"]),
                # EnsureChannelFirstd(keys=["image"]),
                ScaleIntensityRanged(
                    keys=["image"],
                    a_min=0,
                    a_max=255,
                    b_min=0.0,
                    b_max=1.0,
                    clip=True,
                ),
            ],
        )

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        image_id = self.image_ids[idx]
        img = tifffile.imread(self.data_dir / f"{image_id}.tif")

        # Add channel dimension if not present (3D -> 4D: C, D, H, W)
        if img.ndim == 3:
            img = np.expand_dims(img, axis=0)

        # Convert to float for transforms
        img = img.astype(np.float32)

        d = self.transforms(
            {
                "image": img,
            },
        )
        # Use as_tensor to avoid memory copy when possible (shares memory with numpy)
        # Keep as float32 - autocast will handle fp16 conversion during training
        # Note: Tensors are on CPU here; pin_memory in DataLoader handles GPU transfer
        pixel_values = torch.as_tensor(d["image"], dtype=torch.float32)

        return {
            # HF model wrapper expects `pixel_values` in forward signature
            "pixel_values": pixel_values,
            "label_fg_bin": None,  # binary ground truth [0, 1]
            "label_fg": None,  # signed distance field (SDF), used for surface loss
            "label_valid": None,  # valid region mask
            "distance_unsigned": None,  # unsigned distance from foreground (regression target)
            "idx": idx,
            "id": image_id,  # Use actual image ID if available
        }


# =============================================================================
# Data Collator (Same as training)
# =============================================================================
def data_collator(batch):
    """Collate batch samples into tensors, padding to max size in batch.

    Handles variable-sized inputs (256, 320, 384) by padding to the max
    spatial dimensions in the current batch.

    For inference mode (labels are None), only collates pixel_values.
    """
    # Get max spatial dimensions across batch
    pixel_values_list = [x["pixel_values"] for x in batch]

    # Max shape for pixel values (4D: C, D, H, W)
    max_shape_pv = [
        max(pv.shape[i] for pv in pixel_values_list) for i in range(len(pixel_values_list[0].shape))
    ]

    # Pad all samples to max shape
    padded_pixel_values = []

    for sample in batch:
        pv = sample["pixel_values"]  # (C, D, H, W)

        # Calculate padding for pixel values (4D: C, D, H, W)
        padding_pv = []
        for i in range(len(pv.shape) - 1, -1, -1):  # Reverse order for torch.nn.functional.pad
            pad_total = max(0, max_shape_pv[i] - pv.shape[i])
            pad_before = pad_total // 2
            pad_after = pad_total - pad_before
            padding_pv.extend([pad_before, pad_after])

        # Pad pixel values (with constant value 0 for images)
        if any(p > 0 for p in padding_pv):
            pv_padded = torch.nn.functional.pad(pv, padding_pv, mode="constant", value=0)
        else:
            pv_padded = pv

        padded_pixel_values.append(pv_padded)

    return {
        "pixel_values": torch.stack(padded_pixel_values),
        "idx": torch.tensor([x["idx"] for x in batch], dtype=torch.long),
        "id": [x["id"] for x in batch],  # Keep as list of IDs
    }


# =============================================================================
# Inference Function
# =============================================================================
@torch.no_grad()
def run_inference(
    model: TopologyPreservingUNet,
    dataloader: DataLoader,
    device: torch.device,
    roi_size: tuple[int, int, int],
    output_dir: Path,
    use_amp: bool = True,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Run inference on dataset and save predictions.

    Args:
        model: Trained model to use for inference.
        dataloader: DataLoader for inference data.
        device: Device to run inference on.
        roi_size: Patch size for sliding window inference (Z, Y, X).
        output_dir: Directory to save predictions.
        use_amp: Whether to use automatic mixed precision.

    Returns:
        Tuple of:
            - Dictionary mapping sample idx to probability predictions
            - Dictionary mapping sample idx to distance predictions

    """
    model.eval()
    model.to(device)

    # Create subdirectories for each output type
    probs_dir = output_dir / "probabilities"
    distance_dir = output_dir / "distance_preds"
    probs_dir.mkdir(parents=True, exist_ok=True)
    distance_dir.mkdir(parents=True, exist_ok=True)

    all_probs = {}
    all_distance_preds = {}

    print(f"\nRunning inference on {len(dataloader)} batches...")
    print(f"  Device: {device}")
    print(f"  ROI size: {roi_size}")
    print(f"  Mixed precision: {use_amp}")
    print(f"  Output directory: {output_dir}")
    print(f"    - Probabilities: {probs_dir}")
    print(f"    - Distance preds: {distance_dir}")

    for batch_idx, batch in enumerate(tqdm(dataloader, desc="Inference")):
        img = batch["pixel_values"].to(device)  # (B, C, D, H, W)
        indices = batch["idx"].cpu().numpy()
        image_ids = batch["id"]  # List of image IDs

        # Define prediction function for sliding window that returns both outputs
        # We concatenate logits and distance_pred along channel dim for sliding window
        def predict_fn(x):
            outputs = model(pixel_values=x, label_fg_bin=None, label_valid=None, distance_map=None)
            # Concatenate along channel dimension: [B, 2, D, H, W]
            return torch.cat([outputs["logits"], outputs["distance_pred"]], dim=1)

        # Run sliding window inference with mixed precision if enabled
        if use_amp:
            with torch.cuda.amp.autocast():
                combined_output = sliding_window_inference(
                    inputs=img,
                    roi_size=roi_size,
                    sw_batch_size=1,
                    predictor=predict_fn,
                    overlap=0.1,
                    mode="gaussian",
                    device=device,
                )
        else:
            combined_output = sliding_window_inference(
                inputs=img,
                roi_size=roi_size,
                sw_batch_size=1,
                predictor=predict_fn,
                overlap=0.1,
                mode="gaussian",
                device=device,
            )

        # Split the combined output back into logits and distance_pred
        logits = combined_output[:, 0:1, ...]  # (B, 1, D, H, W)
        distance_pred = combined_output[:, 1:2, ...]  # (B, 1, D, H, W)

        # Convert logits to probabilities using sigmoid
        probs = torch.sigmoid(logits).cpu().numpy()  # (B, 1, D, H, W)
        distance_pred_np = distance_pred.cpu().numpy()  # (B, 1, D, H, W)

        # Save predictions for each sample in batch
        batch_size = probs.shape[0]
        for i in range(batch_size):
            idx = indices[i]
            image_id = image_ids[i]
            sample_probs = probs[i, 0]  # Remove channel dim: (D, H, W)
            sample_distance = distance_pred_np[i, 0]  # Remove channel dim: (D, H, W)

            # Store in memory
            all_probs[idx] = sample_probs
            all_distance_preds[idx] = sample_distance

            # Save to disk - use image ID as filename
            probs_path = probs_dir / f"{image_id}.npy"
            distance_path = distance_dir / f"{image_id}.npy"
            np.save(probs_path, sample_probs.astype(np.float16))
            np.save(distance_path, sample_distance.astype(np.float16))

        # Progress update every 10 batches
        if (batch_idx + 1) % 10 == 0:
            print(f"  Processed {(batch_idx + 1) * dataloader.batch_size} samples")

    print(f"\n✓ Inference complete! Saved {len(all_probs)} predictions to {output_dir}")

    return all_probs, all_distance_preds


# =============================================================================
# Configuration Loading (Same as training)
# =============================================================================
def _load_config(experiment: str, config_path: str | None = None) -> DictConfig:
    """Load and merge configuration files.

    Args:
        experiment: Name of the experiment (loads from configs/experiments/<name>.yaml)
        config_path: Optional direct path to config YAML (skips experiment lookup)

    Returns:
        Merged DictConfig with all settings.

    """
    if config_path:
        exp_cfg = OmegaConf.load(config_path)
        # Load globals relative to BASE_DIR
        globals_path = BASE_DIR / "configs" / "globals.yaml"
        if globals_path.exists():
            globals_cfg = OmegaConf.load(str(globals_path))
            cfg = OmegaConf.merge(globals_cfg, exp_cfg)
        else:
            cfg = exp_cfg
    else:
        # Load experiment config
        exp_cfg_path = (
            BASE_DIR
            / "src"
            / "approach"
            / "unetbasic"
            / "configs"
            / "experiments"
            / f"{experiment}.yaml"
        )
        if not exp_cfg_path.exists():
            raise FileNotFoundError(
                f"Experiment config not found: {exp_cfg_path}\n"
                f"Available experiments: {list((exp_cfg_path.parent).glob('*.yaml'))}",
            )

        exp_cfg = OmegaConf.load(str(exp_cfg_path))

        # Load and merge globals
        globals_path = BASE_DIR / "configs" / "globals.yaml"
        if globals_path.exists():
            globals_cfg = OmegaConf.load(str(globals_path))
            cfg = OmegaConf.merge(globals_cfg, exp_cfg)
        else:
            print(
                f"Warning: globals.yaml not found at {globals_path}, using only experiment config",
            )
            cfg = exp_cfg

    if not isinstance(cfg, DictConfig):
        raise TypeError(f"Expected DictConfig, got {type(cfg)}")

    return cfg


# =============================================================================
# Main Entry Point
# =============================================================================
def main() -> None:
    """CLI entry point for inference.

    Loads trained checkpoint, prepares data using same preprocessing as training,
    and generates predictions on validation set.
    """
    global BASE_DIR

    parser = argparse.ArgumentParser(description="Run inference with trained BasicUNet")
    parser.add_argument(
        "--experiment",
        type=str,
        default="exp_v0",
        help="Experiment name (loads config from configs/experiments/<name>.yaml)",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=str(BASE_DIR / "models/dynunetclassreg/jan7/checkpoint-58650"),
        help="Path to trained checkpoint directory (e.g., outputs/monai_swinunetr/checkpoint-1000)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Override: direct path to config YAML (skips experiment lookup)",
    )
    parser.add_argument(
        "--images_dir",
        type=str,
        default=str(BASE_DIR / "scoring/inference_images"),
        help="Images for inference directory for data",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(BASE_DIR / "scoring/temp_dynunetclassreg"),
        help="Output directory for predictions (default: <BASE_DIR>/scoring/temp_dynunetclassreg)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size for inference (default: 1)",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=8,
        help="Number of data loading workers (default: 8)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to run on (default: auto-detect cuda/cpu)",
    )
    args = parser.parse_args()

    # Update BASE_DIR if provided
    IMAGES_DIR = Path(args.images_dir)

    # Load configuration (same as training)
    cfg = _load_config(args.experiment, args.config)
    cfg.approach = "classreg"
    cfg.experiment = args.experiment

    # Setup device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n{'=' * 80}")
    print("BasicUNet Inference")
    print(f"{'=' * 80}")
    print(f"Experiment: {args.experiment}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Device: {device}")
    print(f"Images directory: {IMAGES_DIR}")

    # Setup output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        raise ValueError("Output directory (--output_dir) must be specified")

    # ==========================================================================
    # Load Data (Same as training)
    # ==========================================================================
    print(f"\n{'=' * 80}")
    print("Loading Data")
    print(f"{'=' * 80}")

    # List all image ids from directory
    image_ids = [i.stem for i in sorted(IMAGES_DIR.glob("*.tif"))]

    print(f"Validation samples: {len(image_ids)}")

    # Create dataset (no augmentation for inference)
    patch_size = (
        cfg.data.patch_size.z,
        cfg.data.patch_size.y,
        cfg.data.patch_size.x,
    )
    patch_size = (128, 320, 320)  # Override for inference

    val_dataset = PapyrusDataset(
        data_dir=IMAGES_DIR,
        image_ids=image_ids,
        patch_size=patch_size,
        augmentation=False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=data_collator,
        pin_memory=True,
    )

    print(f"Patch size: {patch_size}")
    print(f"Batch size: {args.batch_size}")
    print(f"Num workers: {args.num_workers}")

    # ==========================================================================
    # Load Model (Same architecture as training)
    # ==========================================================================
    print(f"\n{'=' * 80}")
    print("Loading Model")
    print(f"{'=' * 80}")

    # Extract loss configuration (not used for inference, but needed for model init)
    loss_config = None
    if hasattr(cfg, "loss") and hasattr(cfg.loss, "ThinManifoldLoss"):
        loss_config = OmegaConf.to_container(cfg.loss.ThinManifoldLoss, resolve=True)

    # Initialize model with same architecture as training
    model = TopologyPreservingUNet(
        in_channels=1,
        out_channels=1,
        base_channels=getattr(cfg.model, "base_channels", 32),
        loss_config=loss_config,
    )

    # Load checkpoint
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    # Try loading from HF Trainer checkpoint format
    model_path = checkpoint_path / "pytorch_model.bin"
    if not model_path.exists():
        # Try alternative checkpoint format
        model_path = checkpoint_path / "model.safetensors"
        if not model_path.exists():
            raise FileNotFoundError(
                f"No model found in checkpoint directory.\n"
                f"Looked for: pytorch_model.bin or model.safetensors in {checkpoint_path}",
            )

    print(f"Loading checkpoint from: {model_path}")

    # Load state dict
    if model_path.suffix == ".safetensors":
        from safetensors.torch import load_file

        state_dict = load_file(str(model_path))
    else:
        state_dict = torch.load(model_path, map_location=device)

    # Load weights into model
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    print("✓ Model loaded successfully")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  Trainable: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # ==========================================================================
    # Run Inference
    # ==========================================================================
    print(f"\n{'=' * 80}")
    print("Running Inference")
    print(f"{'=' * 80}")

    use_amp = getattr(cfg.training, "fp16", False)
    probs, distance_preds = run_inference(
        model=model,
        dataloader=val_loader,
        device=device,
        roi_size=patch_size,
        output_dir=output_dir,
        use_amp=use_amp,
    )

    # ==========================================================================
    # Summary
    # ==========================================================================
    print(f"\n{'=' * 80}")
    print("Inference Complete")
    print(f"{'=' * 80}")
    print(f"Total predictions: {len(probs)}")
    print(f"Output directory: {output_dir}")
    print("\nPrediction files saved in subdirectories:")
    print("  - probabilities/: Sigmoid probabilities from logits")
    print("  - distance_preds/: Distance field predictions")
    print("\nEach file is: <image_id>.npy with shape (D, H, W), dtype float16")
    print("\nTo load predictions:")
    print("  probs = np.load('probabilities/3320274.npy')")
    print("  distance = np.load('distance_preds/3320274.npy')")


if __name__ == "__main__":
    main()
