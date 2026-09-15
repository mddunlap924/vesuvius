"""Train 3D UNet for Vesuvius segmentation - Simplified version using standard Trainer.

This version removes the complex MemoryEfficientTrainer in favor of:
- Standard HuggingFace Trainer with built-in DDP/Accelerate support
- Built-in EarlyStoppingCallback from transformers
- Simple TTA wrapper for evaluation
- Memory-efficient metric computation via preprocess_logits_for_metrics

To run with multi-GPU using DDP (recommended over DataParallel):
    accelerate launch --num_processes=2 train_simplified.py --config your_config.yaml

Or single GPU:
    python train_simplified.py --config your_config.yaml
"""

import argparse
import gc
import os
import shutil
import sys
import time
import warnings
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TypeAlias, cast

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import torch.nn.functional as F
from monai.inferers import sliding_window_inference
from omegaconf import DictConfig, OmegaConf
from sklearn.model_selection import train_test_split
from transformers import EarlyStoppingCallback, Trainer, TrainerCallback, TrainingArguments

from approach.unetbasic.data.dataset import PapyrusDataset

# from approach.unetbasic.metrics.kaggle_metrics import compute_average_metrics
from approach.unetbasic.models.unet import TopologyPreservingUNet, WrapperBasicUnet
from approach.unetbasic.utils.logger import ExperimentLogger
from approach.unetbasic.utils.loss_weight_schedule import compute_scheduled_weights

warnings.filterwarnings("ignore", category=UserWarning)

# Default BASE_DIR (project root) - can be overridden via --base_dir argument
DEFAULT_BASE_DIR: Path = Path(__file__).resolve().parent.parent.parent.parent
# Override with the BASE_DIR env var or the --base_dir argument
BASE_DIR: Path = DEFAULT_BASE_DIR  # Will be updated in main() if --base_dir is provided
# BASE_DIR: Path = Path(os.getenv("BASE_DIR"))


# =============================================================================
# Simple Console Logger
# =============================================================================
class ConsoleLogger(ExperimentLogger):
    """Simple logger that prints to console when no wandb logger is provided."""

    def __init__(self) -> None:
        self.run = None
        self.output_dir = None

    def log_metrics(self, metrics: dict, step=None, console=True) -> None:
        timestamp = datetime.now(tz=UTC).isoformat(timespec="seconds")
        metrics_str = ", ".join(
            f"{k}={v:.4f}" if isinstance(v, (int, float)) else f"{k}={v}"
            for k, v in metrics.items()
        )
        sys.stdout.write(f"[{timestamp}] step={step} | {metrics_str}\n")
        sys.stdout.flush()


# =============================================================================
# W&B Config Logging Callback
# =============================================================================
class WandbConfigCallback(TrainerCallback):
    """Callback to log full experiment config to W&B.

    The HuggingFace Trainer's WandbCallback only logs TrainingArguments,
    missing custom config values like model.unet3d.init_features, losses, etc.
    This callback updates the W&B config with the full experiment configuration.
    """

    def __init__(self, cfg: DictConfig):
        self.cfg = cfg

    def on_train_begin(self, args, state, control, **kwargs):
        """Log config to W&B at the start of training (after W&B is initialized)."""
        try:
            import wandb

            if wandb.run is not None:
                # Convert OmegaConf to a plain dict for W&B
                config_dict = OmegaConf.to_container(self.cfg, resolve=True)
                # Update W&B config with our full experiment config
                wandb.config.update(config_dict, allow_val_change=True)
        except ImportError:
            pass  # wandb not installed
        except Exception as e:
            # Don't fail training if config logging fails
            sys.stdout.write(f"[WARNING] Failed to log config to W&B: {e}\n")


# =============================================================================
# Loss Weight Scheduler Callback
# =============================================================================
class LossWeightSchedulerCallback(TrainerCallback):
    """Callback to dynamically update loss weights during training.

    Implements a curriculum learning approach where basic segmentation losses
    (Dice, BCE, Tversky) remain stable, while topology/connectivity losses
    ramp up gradually after the model learns the basic sheet structure.

    Reads schedule from cfg.loss_weight_schedule with per-weight ramp specs:
        ramps:
            weight_name:
                start_epoch: 0
                end_epoch: 40
                start_value: 0.0
                end_value: 0.15

    Computes linear interpolation between start/end values based on current epoch.
    Only updates scheduled weights; unscheduled weights retain their initial values.
    """

    def __init__(
        self,
        schedule_cfg: DictConfig,
        model: torch.nn.Module,
        logger: ExperimentLogger | None = None,
    ):
        """Initialize the loss weight scheduler.

        Args:
            schedule_cfg: Configuration dict with 'enabled' flag and 'ramps' section
            model: The model containing the loss function to update
            logger: Optional experiment logger for logging weight updates

        """
        self.schedule_cfg = schedule_cfg
        self.model = model
        self.logger = logger
        self._last_applied_epoch = -1

        # Validate schedule configuration
        if not schedule_cfg.get("enabled", False):
            logging.warning("LossWeightSchedulerCallback created but schedule is disabled")

        if "ramps" not in schedule_cfg:
            msg = "loss_weight_schedule.ramps section missing from config"
            raise ValueError(msg)

    def _get_loss_fn(self):
        """Get the loss function, unwrapping DDP/DataParallel if needed."""
        model = self.model
        # Unwrap DDP wrapper if using distributed training
        if hasattr(model, "module"):
            model = model.module
        return getattr(model, "loss_fn", None)

    def on_epoch_begin(self, args, state, control, **kwargs):
        """Update loss weights at the start of each epoch.

        Args:
            args: TrainingArguments
            state: TrainerState with current epoch information
            control: TrainerControl
            **kwargs: Additional arguments

        """
        # Get current epoch (may be fractional for sub-epoch checkpoints)
        current_epoch = float(state.epoch) if state.epoch is not None else 0.0

        # Skip if we've already applied weights for this epoch
        if int(current_epoch) == self._last_applied_epoch:
            return

        # Get the loss function
        loss_fn = self._get_loss_fn()
        if loss_fn is None:
            logging.warning("Could not find loss_fn on model for weight scheduling")
            return

        # Compute scheduled weights for current epoch
        scheduled_weights = compute_scheduled_weights(
            self.schedule_cfg,
            epoch=current_epoch,
        )

        if not scheduled_weights:
            # No weights scheduled for update at this epoch
            return

        # Apply weights to loss function
        try:
            updated_weights = loss_fn.update_weights(scheduled_weights)
            self._last_applied_epoch = int(current_epoch)

            # Log weight updates (only from main process to avoid duplicate logs)
            if _is_main_process():
                weight_str = ", ".join(f"{k}={v:.4f}" for k, v in scheduled_weights.items())
                log_msg = f"Epoch {int(current_epoch)}: Loss weights updated - {weight_str}"

                if self.logger:
                    self.logger.log_metrics(
                        {
                            "message": log_msg,
                            **{f"loss/{k}": v for k, v in updated_weights.items()},
                        },
                        step=state.global_step,
                    )
                else:
                    print(log_msg)

                # Also log to W&B if available
                try:
                    import wandb

                    if wandb.run is not None:
                        wandb.log(
                            {f"loss_weights/{k}": v for k, v in updated_weights.items()},
                            step=state.global_step,
                        )
                except ImportError:
                    pass

        except Exception as e:
            logging.error(f"Failed to update loss weights at epoch {current_epoch}: {e}")


# =============================================================================
# UNET Trainer
# =============================================================================
class UNETTrainer(Trainer):
    def __init__(self, *args, roi_size=(32, 320, 320), **kwargs):
        super().__init__(*args, **kwargs)
        self.roi_size = roi_size

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """Override to filter out non-model keys before forward pass.

        Supports both WrapperBasicUnet (single head) and TopologyPreservingUNet (dual heads).
        """
        # Extract model inputs - works for both model types
        model_inputs = {
            "pixel_values": inputs["pixel_values"],
            "label_fg_bin": inputs["label_fg_bin"],
            "label_valid": inputs["label_valid"],
            "distance_map": inputs["label_fg"],  # signed distance field from transforms
        }

        # Add distance_unsigned for TopologyPreservingUNet
        if "distance_unsigned" in inputs:
            model_inputs["distance_unsigned"] = inputs["distance_unsigned"]

        outputs = model(**model_inputs)
        loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
        return (loss, outputs) if return_outputs else loss

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        img = inputs["pixel_values"]  # shape: (B, C, D, H, W)
        label_fg_bin = inputs["label_fg_bin"]  # binary ground truth
        label_valid = inputs["label_valid"]

        # Create a wrapper that returns just the logits tensor
        # sliding_window_inference expects predictor to return a tensor, not a dict
        def predict_fn(x):
            out = model(pixel_values=x)
            # Both WrapperBasicUnet and TopologyPreservingUNet return "logits"
            return out["logits"] if isinstance(out, dict) else out

        # Determine if mixed precision is enabled
        use_amp = self.args.fp16 or getattr(self.args, "bf16", False)

        with torch.no_grad():
            # Use autocast for fp16/bf16 inference - matches training precision
            with torch.autocast(device_type="cuda", enabled=use_amp):
                logits = sliding_window_inference(
                    inputs=img,
                    roi_size=self.roi_size,
                    sw_batch_size=1,
                    predictor=predict_fn,
                    overlap=0.50,
                )

                # Compute eval loss using the model's loss function
                # Access the underlying model (handles DDP wrapping)
                base_model = model.module if hasattr(model, "module") else model
                distance_map = inputs["label_fg"]  # signed distance field

                # Handle different model types
                # TopologyPreservingUNet has loss_fn with classification_loss attribute
                if hasattr(base_model, "loss_fn") and hasattr(
                    base_model.loss_fn,
                    "classification_loss",
                ):
                    # TopologyPreservingUNet - use classification_loss for eval
                    # (don't compute distance regression during sliding window inference)
                    loss = base_model.loss_fn.classification_loss(
                        logits=logits,
                        label_fg_bin=label_fg_bin,
                        label_valid=label_valid,
                        distance_map=distance_map,
                    )
                else:
                    # WrapperBasicUnet with ThinManifoldLoss
                    loss = base_model.loss_fn(
                        logits=logits,
                        label_fg_bin=label_fg_bin,
                        label_valid=label_valid,
                        distance_map=distance_map,
                    )

        # preprocess_logits_for_metrics computes all metrics from logits+labels,
        # returning fixed-size stats. Return dummy labels to avoid concatenation
        # errors when validation samples have different sizes.
        batch_size = logits.shape[0]
        dummy_labels = torch.zeros(batch_size, 1, device=logits.device)

        return (loss, (logits, label_fg_bin, label_valid), dummy_labels)


# =============================================================================
# Data Collator
# =============================================================================
def data_collator(batch):
    return {
        "pixel_values": torch.stack([x["pixel_values"] for x in batch]),
        "label_fg_bin": torch.stack([x["label_fg_bin"] for x in batch]),
        "label_fg": torch.stack([x["label_fg"] for x in batch]),  # signed distance field (SDF)
        "label_valid": torch.stack([x["label_valid"] for x in batch]),
        "distance_unsigned": torch.stack(
            [x["distance_unsigned"] for x in batch],
        ),  # unsigned distance (regression target)
        "idx": torch.tensor([x["idx"] for x in batch], dtype=torch.long),
    }


# =============================================================================
# Metric Functions
# =============================================================================
def preprocess_logits_for_metrics(logits, labels):
    """Convert logits and labels to aggregated statistics for efficient metric computation.

    This reduces memory by computing statistics immediately rather than accumulating
    large tensors. Returns a small tensor of statistics per sample.

    Statistics format (10 values per sample):
        [tp, fp, fn, tn, n_valid, sum_pred, sum_label, proxy_surface_dice, proxy_topo, proxy_voi]

    Memory optimized: minimizes float() conversions, uses preallocated tensors, bool ops.
    """
    # Unpack the tuple: logits contains (actual_logits, label_fg, label_valid)
    if isinstance(logits, tuple):
        actual_logits, label_fg, label_valid = logits
    else:
        actual_logits = logits
        # Fallback - shouldn't happen with new pipeline
        label_fg = labels
        label_valid = torch.ones_like(labels)

    # Handle dict output from model
    if isinstance(actual_logits, dict):
        actual_logits = actual_logits.get("logits", actual_logits)

    # Get predictions: sigmoid threshold for single-channel binary output
    # Use no_grad to avoid storing intermediate activations
    with torch.no_grad():
        probs = torch.sigmoid(actual_logits)  # [B, 1, D, H, W]
        preds = (probs > 0.5).squeeze(1)  # [B, D, H, W] - keep as bool

    # Handle label shapes - may have channel dim
    if label_fg.dim() == actual_logits.dim():
        label_fg = label_fg.squeeze(1)  # [B, 1, D, H, W] -> [B, D, H, W]
    if label_valid.dim() == actual_logits.dim():
        label_valid = label_valid.squeeze(1)  # [B, 1, D, H, W] -> [B, D, H, W]

    batch_size = preds.shape[0]
    device = actual_logits.device

    # Preallocate stats tensor to avoid per-sample tensor creation
    stats_tensor = torch.zeros(batch_size, 10, device=device, dtype=torch.float32)

    for i in range(batch_size):
        pred = preds[i]  # [D, H, W] - bool tensor
        fg = label_fg[i]  # [D, H, W] - float tensor (0 or 1)
        valid = label_valid[i]  # [D, H, W] - float tensor (0 or 1)

        # Handle shape mismatch between predictions and labels
        if pred.shape != fg.shape:
            pred = (
                F.interpolate(
                    pred.float().unsqueeze(0).unsqueeze(0),
                    size=fg.shape,
                    mode="nearest",
                )
                .squeeze(0)
                .squeeze(0)
                > 0.5
            )  # Back to bool [D, H, W]

        # Create mask for valid (non-padding) regions
        valid_mask = valid > 0.5  # bool mask

        # Compute confusion matrix directly on bool tensors (no float conversion)
        pred_valid = pred[valid_mask]  # bool tensor
        label_valid = fg[valid_mask] > 0.5  # bool tensor

        tp = (pred_valid & label_valid).sum()
        fp = (pred_valid & ~label_valid).sum()
        fn = (~pred_valid & label_valid).sum()
        tn = (~pred_valid & ~label_valid).sum()
        n_valid = valid_mask.sum()
        sum_pred = pred_valid.sum()
        sum_label = label_valid.sum()

        # Convert to float only for final metrics computation
        tp_f, fp_f, fn_f = tp.float(), fp.float(), fn.float()

        # Compute simple proxy metrics (fast approximations)
        dice = (2 * tp_f) / (2 * tp_f + fp_f + fn_f + 1e-6)

        # Proxy Topology: measure connectivity consistency
        # Cache float conversion (do once, not twice)
        pred_float = pred.float()
        label_float = (fg > 0.5).float()
        pred_edges = torch.abs(pred_float[1:] - pred_float[:-1]).sum()
        label_edges = torch.abs(label_float[1:] - label_float[:-1]).sum()
        edge_ratio = 1.0 - torch.abs(pred_edges - label_edges) / (label_edges + pred_edges + 1e-6)

        # Proxy VOI: Use F1 as approximation
        precision = tp_f / (tp_f + fp_f + 1e-6)
        recall = tp_f / (tp_f + fn_f + 1e-6)
        f1 = 2 * precision * recall / (precision + recall + 1e-6)

        # Fill preallocated tensor (avoids creating new tensor per sample)
        stats_tensor[i, 0] = tp
        stats_tensor[i, 1] = fp
        stats_tensor[i, 2] = fn
        stats_tensor[i, 3] = tn
        stats_tensor[i, 4] = n_valid
        stats_tensor[i, 5] = sum_pred
        stats_tensor[i, 6] = sum_label
        stats_tensor[i, 7] = dice
        stats_tensor[i, 8] = edge_ratio
        stats_tensor[i, 9] = f1

    return stats_tensor  # [B, 10]


def create_compute_metrics_fn():
    """Factory to create compute_metrics for proxy metrics during training.

    Returns a function that computes fast GPU-based proxy metrics.
    Expensive Kaggle metrics are computed once after training completes.
    """

    def compute_metrics(eval_pred):
        """Compute metrics from aggregated statistics."""
        predictions, labels = eval_pred
        stats = (
            torch.from_numpy(predictions)
            if not isinstance(predictions, torch.Tensor)
            else predictions
        )

        # Aggregate standard statistics (indices 0-6)
        tp = stats[:, 0].sum()
        fp = stats[:, 1].sum()
        fn = stats[:, 2].sum()
        tn = stats[:, 3].sum()
        n_valid = stats[:, 4].sum()
        sum_pred = stats[:, 5].sum()
        sum_label = stats[:, 6].sum()

        # Proxy metrics (indices 7-9) - average across samples
        proxy_surface_dice = stats[:, 7].mean()
        proxy_topo = stats[:, 8].mean()
        proxy_voi = stats[:, 9].mean()

        # Combined proxy score (same weights as Kaggle: topo=0.30, surface=0.35, voi=0.35)
        proxy_kaggle = 0.30 * proxy_topo + 0.35 * proxy_surface_dice + 0.35 * proxy_voi

        # Compute standard metrics
        precision = tp / (tp + fp + 1e-6)
        recall = tp / (tp + fn + 1e-6)
        f1 = 2 * precision * recall / (precision + recall + 1e-6)
        accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-6)
        dice = (2 * tp) / (2 * tp + fp + fn + 1e-6)
        avg_pred = sum_pred / (n_valid + 1e-6)
        avg_label = sum_label / (n_valid + 1e-6)

        return {
            "dice": float(dice),
            "accuracy": float(accuracy),
            "f1": float(f1),
            "precision": float(precision),
            "recall": float(recall),
            "avg_pred": float(avg_pred),
            "avg_label": float(avg_label),
            # Proxy metrics (fast GPU-based approximations)
            "proxy_surface_dice": float(proxy_surface_dice),
            "proxy_topo": float(proxy_topo),
            "proxy_voi": float(proxy_voi),
            "proxy_kaggle": float(proxy_kaggle),
        }

    return compute_metrics


# =============================================================================
# Utility Functions
# =============================================================================
def convert_epoch_pct_to_steps(
    epoch_pct: float,
    dataset_size: int,
    per_device_batch_size: int,
    gradient_accumulation_steps: int = 1,
    num_devices: int = 1,
) -> int:
    """Convert epoch percentage to training steps."""
    effective_batch_size = per_device_batch_size * num_devices * gradient_accumulation_steps
    steps_per_epoch = dataset_size / effective_batch_size
    return max(1, int(steps_per_epoch * epoch_pct))


# =============================================================================
# Main Training Function
# =============================================================================
def _is_main_process() -> bool:
    """Check if this is the main process in distributed training."""
    if dist.is_initialized():
        return dist.get_rank() == 0
    # Check environment variables set by accelerate/torchrun before dist.init
    local_rank = os.environ.get("LOCAL_RANK")
    if local_rank is not None:
        return int(local_rank) == 0
    return True


def _barrier() -> None:
    """Synchronize all processes if distributed training is active."""
    if dist.is_initialized():
        dist.barrier()
    elif os.environ.get("LOCAL_RANK") is not None:
        # When using accelerate but dist not yet initialized, we need to wait
        # Use a simple sleep to give main process time to complete directory operations
        time.sleep(0.5)


def run_experiment(cfg: DictConfig, *, logger: ExperimentLogger | None = None) -> dict:
    """Run 3D UNet training experiment using standard Trainer with DDP support."""
    if logger is None:
        logger = ConsoleLogger()

    # Log GPU info (only from main process to avoid duplicate logs)
    if _is_main_process():
        logger.log_metrics({"message": f"GPUs available: {torch.cuda.device_count()}"}, step=0)
        for i in range(torch.cuda.device_count()):
            logger.log_metrics({"message": f"GPU {i}: {torch.cuda.get_device_name(i)}"}, step=0)

    # ==========================================================================
    # Data Preparation
    # ==========================================================================
    # List of image/label ID pairs
    data_dir = BASE_DIR / "data/kaggle_converted"
    image_ids = [p.stem for p in (data_dir / "train_images").glob("*.npy")]

    # Train test split
    train_image_ids, val_image_ids = train_test_split(
        image_ids,
        test_size=cfg.test_size,
        random_state=42,
    )
    logger.log_metrics(
        {
            "message": (
                f"Train/Test Split {len(train_image_ids)} images for training, "
                f"{len(val_image_ids)} images for validation"
            ),
        },
        step=0,
    )

    # Debug mode downsampling - downsample data_df BEFORE patch generation for speed
    if getattr(cfg, "debug", {}).get("enabled", False):
        factor = getattr(cfg.debug, "downsample_factor", 0.1)
        train_image_ids = train_image_ids[: max(1, int(len(train_image_ids) * factor))]
        val_image_ids = val_image_ids[: max(1, int(len(val_image_ids) * factor))]

    # Train and validation ct images
    logger.log_metrics(
        {
            "message": (
                f"Using {len(train_image_ids)} images for training, "
                f"{len(val_image_ids)} images for validation"
            ),
        },
        step=0,
    )
    # Datasets
    patch_size = (
        cfg.data.patch_size.z,
        cfg.data.patch_size.y,
        cfg.data.patch_size.x,
    )

    train_dataset = PapyrusDataset(
        data_dir=data_dir,
        image_ids=train_image_ids,
        augmentation_config=getattr(cfg, "augmentation", None),
        patch_size=patch_size,
        augmentation=True,
    )
    val_dataset = PapyrusDataset(
        data_dir=data_dir,
        image_ids=val_image_ids,
        patch_size=patch_size,
        augmentation=False,
    )

    # Validate dataset output shapes match config expectations
    expected_input_shape = (1, *patch_size)
    expected_label_shape = (
        1,
        *patch_size,
    )  # label_fg_bin, label_fg, and label_valid have channel dim

    train_sample = train_dataset[0]
    val_sample = val_dataset[0]

    assert train_sample["pixel_values"].shape == expected_input_shape, (
        f"Train input shape mismatch: got {train_sample['pixel_values'].shape}, "
        f"expected {expected_input_shape}"
    )
    assert train_sample["label_fg_bin"].shape == expected_label_shape, (
        f"Train label_fg_bin shape mismatch: got {train_sample['label_fg_bin'].shape}, "
        f"expected {expected_label_shape}"
    )
    assert train_sample["label_fg"].shape == expected_label_shape, (
        f"Train label_fg (distance map) shape mismatch: got {train_sample['label_fg'].shape}, "
        f"expected {expected_label_shape}"
    )
    assert train_sample["label_valid"].shape == expected_label_shape, (
        f"Train label_valid shape mismatch: got {train_sample['label_valid'].shape}, "
        f"expected {expected_label_shape}"
    )

    logger.log_metrics(
        {
            "message": f"Train: {len(train_dataset)} samples, Val: {len(val_dataset)} samples",
        },
        step=0,
    )

    # ==========================================================================
    # Model and Loss Configuration
    # ==========================================================================
    # Uses TopologyPreservingUNet with TopologyPreservingLoss (dual-head model).
    # - Binary head: segmentation with Dice/BCE/Tversky/etc losses
    # - Distance head: SDF regression for topological continuity
    #
    # Configuration structure:
    # - cfg.loss: Contains hyperparameters (tversky_alpha, topo_kernel_sizes, etc.)
    # - cfg.loss_weight_schedule: Controls component weights dynamically during training
    # ==========================================================================

    # Build loss config from cfg.loss (hyperparameters only, no weights)
    loss_config = {}
    if hasattr(cfg, "loss"):
        loss_config = OmegaConf.to_container(cfg.loss, resolve=True)
        # Convert list to tuple for topo_kernel_sizes if present
        if isinstance(loss_config.get("topo_kernel_sizes"), list):
            loss_config["topo_kernel_sizes"] = tuple(loss_config["topo_kernel_sizes"])

    logger.log_metrics(
        {"message": f"Loss hyperparameters: {loss_config}"},
        step=0,
    )

    model = TopologyPreservingUNet(
        in_channels=1,
        out_channels=1,
        base_channels=getattr(cfg.model, "base_channels", 32),
        loss_config=loss_config,
    )

    # Load pretrained weights if specified
    if cfg.model.use_pretrained.apply:
        pretrained_path = BASE_DIR / cfg.model.use_pretrained.path

        # Handle checkpoint directory (HF Trainer format)
        if pretrained_path.is_dir():
            # Check for model.safetensors (HuggingFace checkpoint format)
            safetensors_path = pretrained_path / "model.safetensors"
            pytorch_path = pretrained_path / "pytorch_model.bin"

            if safetensors_path.exists():
                # Load from safetensors format
                from safetensors.torch import load_file

                state_dict = load_file(str(safetensors_path))
                model.load_state_dict(state_dict)
                logger.log_metrics(
                    {
                        "message": f"Loaded pretrained model weights from {safetensors_path}",
                    },
                    step=0,
                )
            elif pytorch_path.exists():
                # Fallback: load from pytorch_model.bin
                state_dict = torch.load(pytorch_path, map_location="cpu")
                model.load_state_dict(state_dict)
                logger.log_metrics(
                    {
                        "message": f"Loaded pretrained model weights from {pytorch_path}",
                    },
                    step=0,
                )
            else:
                raise FileNotFoundError(
                    f"No model weights found in checkpoint directory {pretrained_path}. "
                    f"Expected either model.safetensors or pytorch_model.bin",
                )
        else:
            # Single file path (legacy support)
            state_dict = torch.load(pretrained_path, map_location="cpu")
            model.load_state_dict(state_dict)
            logger.log_metrics(
                {
                    "message": f"Loaded pretrained model weights from {pretrained_path}",
                },
                step=0,
            )
    else:
        logger.log_metrics(
            {
                "message": "Training model from scratch (no pretrained weights)",
            },
            step=0,
        )
    # ==========================================================================
    # Training Arguments
    # ==========================================================================
    model_type = "classreg"
    output_dir = str(BASE_DIR / f"outputs/{model_type}")
    logging_dir = str(BASE_DIR / f"logs/{model_type}")

    # Training parameters
    per_device_batch = getattr(cfg.training, "per_device_train_batch_size", 1)
    grad_accum = getattr(cfg.training, "gradient_accumulation_steps", 1)
    num_devices = torch.cuda.device_count() if torch.cuda.is_available() else 1

    # Convert epoch percentages to steps
    logging_steps = convert_epoch_pct_to_steps(
        getattr(cfg.training, "logging_epoch_pct", 0.05),
        len(train_dataset),
        per_device_batch,
        grad_accum,
        num_devices,
    )
    save_steps = convert_epoch_pct_to_steps(
        getattr(cfg.training, "save_epoch_pct", 0.5),
        len(train_dataset),
        per_device_batch,
        grad_accum,
        num_devices,
    )
    eval_steps = convert_epoch_pct_to_steps(
        getattr(cfg.training, "eval_epoch_pct", 0.5),
        len(train_dataset),
        per_device_batch,
        grad_accum,
        num_devices,
    )
    # When load_best_model_at_end is enabled, save_steps must be a multiple of eval_steps
    early_stopping_enabled = getattr(cfg.training, "early_stopping_enabled", False)
    if early_stopping_enabled and save_steps % eval_steps != 0:
        # Round save_steps up to the nearest multiple of eval_steps
        save_steps = ((save_steps // eval_steps) + 1) * eval_steps
    warmup_steps = convert_epoch_pct_to_steps(
        getattr(cfg.training, "warmup_epoch_pct", 0.0),
        len(train_dataset),
        per_device_batch,
        grad_accum,
        num_devices,
    )

    # W&B configuration
    wandb_mode = getattr(cfg.wandb, "mode", "disabled")
    wandb_project = getattr(cfg.wandb, "project_name", None)
    wandb_entity = getattr(cfg.wandb, "entity", None) or os.getenv("WANDB_ENTITY")
    report_to = ["wandb"] if (wandb_mode == "online" and wandb_project and wandb_entity) else []

    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=per_device_batch,
        per_device_eval_batch_size=getattr(cfg.training, "per_device_eval_batch_size", 1),
        num_train_epochs=getattr(cfg.training, "epochs", 1),
        eval_strategy="steps",
        save_strategy="steps",
        logging_dir=logging_dir,
        logging_steps=logging_steps,
        save_steps=save_steps,
        eval_steps=eval_steps,
        learning_rate=getattr(cfg.training, "learning_rate", 1e-4),
        fp16=getattr(cfg.training, "fp16", True),
        save_total_limit=2,
        report_to=report_to,
        weight_decay=getattr(cfg.training, "weight_decay", 0.01),
        warmup_steps=warmup_steps,
        lr_scheduler_type=getattr(cfg.training, "lr_scheduler_type", "linear"),
        # Data loading
        dataloader_num_workers=getattr(cfg.training, "num_workers", 2),
        dataloader_prefetch_factor=getattr(cfg.training, "prefetch_factor", 2),
        dataloader_pin_memory=True,
        # MEMORY: persistent_workers=False allows worker memory to be freed between epochs
        # This trades some speed for much lower RAM usage
        dataloader_persistent_workers=getattr(cfg.training, "persistent_workers", False),
        gradient_accumulation_steps=grad_accum,
        # Memory optimization: process eval predictions immediately (1 = no accumulation)
        # Higher values accumulate more batches before processing, using more RAM
        eval_accumulation_steps=getattr(cfg.training, "eval_accumulation_steps", 1),
        # For early stopping - use eval_dice since prediction_step doesn't compute loss
        load_best_model_at_end=getattr(cfg.training, "early_stopping_enabled", True),
        metric_for_best_model="eval_dice",
        greater_is_better=True,  # Higher dice is better
        ddp_find_unused_parameters=False,
        remove_unused_columns=False,
        gradient_checkpointing=getattr(cfg.training, "gradient_checkpointing", False),
        # DynUNet has shared tensors between skip_layers and encoder/decoder blocks,
        # which safetensors doesn't support by default. Use standard PyTorch saving.
        save_safetensors=False,
    )

    # ==========================================================================
    # Create Trainer
    # ==========================================================================
    # Create Trainer
    trainer = UNETTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        compute_metrics=create_compute_metrics_fn(),
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        roi_size=patch_size,  # Use config patch_size for sliding window inference
    )

    # Add W&B config callback to log full experiment config (not just TrainingArguments)
    if report_to and "wandb" in report_to:
        trainer.add_callback(WandbConfigCallback(cfg))

    # Add early stopping callback if enabled
    if getattr(cfg.training, "early_stopping_enabled", False):
        patience = getattr(cfg.training, "early_stopping_patience", 3)
        trainer.add_callback(EarlyStoppingCallback(early_stopping_patience=patience))
        logger.log_metrics({"message": f"Early stopping enabled with patience={patience}"}, step=0)

    # Add loss weight scheduler callback if enabled
    if hasattr(cfg, "loss_weight_schedule") and cfg.loss_weight_schedule.get("enabled", False):
        loss_schedule = cfg.loss_weight_schedule
        trainer.add_callback(LossWeightSchedulerCallback(loss_schedule, model, logger))
        if _is_main_process():
            # Log schedule configuration at start
            ramps_summary = []
            for weight_name, ramp_cfg in loss_schedule.get("ramps", {}).items():
                ramps_summary.append(
                    f"{weight_name}: {ramp_cfg.get('start_value', 0):.3f}@{ramp_cfg.get('start_epoch', 0)} "
                    f"-> {ramp_cfg.get('end_value', 0):.3f}@{ramp_cfg.get('end_epoch', 0)}",
                )
            schedule_msg = "Loss weight schedule: " + "; ".join(ramps_summary)
            logger.log_metrics({"message": schedule_msg}, step=0)

    logger.log_metrics(
        {
            "message": f"Training with batch={per_device_batch} x {num_devices} GPUs x {grad_accum} accum",
        },
        step=0,
    )

    # ==========================================================================
    # Train and Evaluate
    # ==========================================================================
    trainer.train()
    logger.log_metrics({"message": "Training complete. Running final evaluation..."}, step=0)
    return


def _load_config(experiment: str, config_path: str | None = None) -> DictConfig:
    """Load and merge configuration files.

    Args:
        experiment: Name of the experiment (loads from configs/experiments/<name>.yaml)
        config_path: Optional direct path to config YAML (skips experiment lookup)

    Returns:
        Merged DictConfig with all settings.

    """
    if config_path:
        if not Path(config_path).exists():
            msg = f"Config not found: {config_path}"
            raise FileNotFoundError(msg)
        cfg = OmegaConf.load(config_path)
    else:
        # Load globals.yaml as base config
        globals_path = Path(os.getenv("BASE_DIR", str(DEFAULT_BASE_DIR))) / "configs/globals.yaml"
        if globals_path.exists():
            cfg = OmegaConf.load(globals_path)
            OmegaConf.set_struct(cfg, False)  # noqa: FBT003

            # Load referenced default configs (data, wandb)
            data_cfg_path = (
                Path(os.getenv("BASE_DIR", str(DEFAULT_BASE_DIR))) / "configs/data/base.yaml"
            )
            if data_cfg_path.exists():
                data_cfg = OmegaConf.load(data_cfg_path)
                cfg = OmegaConf.merge(cfg, {"data": data_cfg})

            wandb_cfg_path = (
                Path(os.getenv("BASE_DIR", str(DEFAULT_BASE_DIR))) / "configs/wandb/wandb.yaml"
            )
            if wandb_cfg_path.exists():
                wandb_cfg = OmegaConf.load(wandb_cfg_path)
                cfg = OmegaConf.merge(cfg, {"wandb": wandb_cfg})
        else:
            cfg = OmegaConf.create({})
            OmegaConf.set_struct(cfg, False)  # noqa: FBT003

        # Load and merge experiment config
        exp_path = (
            Path(os.getenv("BASE_DIR", str(DEFAULT_BASE_DIR)))
            / f"src/approach/unetbasic/configs/experiments/{experiment}.yaml"
        )
        if not exp_path.exists():
            msg = f"Experiment config not found: {exp_path}"
            raise FileNotFoundError(msg)

        exp_cfg = OmegaConf.load(exp_path)
        cfg = OmegaConf.merge(cfg, exp_cfg)

    if not isinstance(cfg, DictConfig):
        cfg = DictConfig(cfg)

    return cfg


def _setup_wandb_env(cfg: DictConfig, wandb_dir: Path) -> None:
    """Configure W&B via environment variables (Trainer handles init on main process)."""
    wandb_mode = getattr(cfg.wandb, "mode", "offline") if hasattr(cfg, "wandb") else "offline"
    wandb_project = getattr(cfg.wandb, "project_name", None) if hasattr(cfg, "wandb") else None
    wandb_entity = (
        getattr(cfg.wandb, "entity", None) if hasattr(cfg, "wandb") else None
    ) or os.getenv("WANDB_ENTITY")

    if wandb_mode == "online" and wandb_project:
        os.environ["WANDB_PROJECT"] = wandb_project
        os.environ["WANDB_DIR"] = str(wandb_dir.absolute())
        os.environ["WANDB_NAME"] = f"{cfg.approach}_{cfg.experiment}"
        if wandb_entity:
            os.environ["WANDB_ENTITY"] = wandb_entity
    elif wandb_mode == "disabled":
        os.environ["WANDB_MODE"] = "disabled"


def main() -> None:
    """CLI entry point with full config loading and W&B setup.

    This is the main entry point for training. It handles:
    - Loading and merging configs (globals.yaml + experiment config)
    - Setting up W&B environment variables (Trainer handles init on main process only)
    - Creating output directories
    - Running the experiment

    Usage:
        Single GPU:  python src/approach/monaiswin/train.py --experiment exp_v0
        Multi-GPU:   accelerate launch --num_processes=2 train.py --experiment exp_v0
    """
    global BASE_DIR
    parser = argparse.ArgumentParser(description="Train 3D UNet for Vesuvius segmentation")
    parser.add_argument(
        "--experiment",
        type=str,
        default="exp_v0",
        help="Experiment name (loads from configs/experiments/<name>.yaml)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Override: direct path to config YAML (skips experiment lookup)",
    )
    parser.add_argument(
        "--base_dir",
        type=str,
        default=str(DEFAULT_BASE_DIR),
        help=f"Base directory for data and configs (default: {DEFAULT_BASE_DIR})",
    )
    args = parser.parse_args()

    # Update BASE_DIR if provided via command line
    BASE_DIR = Path(args.base_dir)

    # Load configuration
    cfg = _load_config(args.experiment, args.config)
    cfg.approach = "unetbasic"
    cfg.experiment = args.experiment

    # Setup output directories
    output_root = Path(cfg.get("output_dir", "outputs")) / cfg.approach / cfg.experiment
    results_dir = output_root / "results"
    wandb_dir = output_root / "wandb"

    if _is_main_process():
        results_dir.mkdir(parents=True, exist_ok=True)
        wandb_dir.mkdir(parents=True, exist_ok=True)
    _barrier()

    cfg.directories = {
        "results_dir": str(results_dir.absolute()),
        "wandb_dir": str(wandb_dir.absolute()),
        "output_root": str(output_root.absolute()),
    }

    # Setup W&B via env vars (Trainer's WandbCallback uses these)
    _setup_wandb_env(cfg, wandb_dir)

    # Run training
    logger = ConsoleLogger()
    results = run_experiment(cfg, logger=logger)

    if _is_main_process():
        logger.log_metrics({"message": "Training complete!"}, step=0)
        # logger.log_metrics(results, step=0)


if __name__ == "__main__":
    main()
