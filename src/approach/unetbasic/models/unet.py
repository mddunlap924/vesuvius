"""UNet wrappers for Hugging Face Trainer integration.

Includes:
- WrapperBasicUnet: Original single-head binary segmentation
- TopologyPreservingUNet: Dual-head model with binary + distance regression outputs
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from monai.networks.blocks import UnetOutBlock
from monai.networks.nets import BasicUnet, DynUNet
from torch import nn
from torch.nn import functional

from ..utils.losses import ThinManifoldLoss, TopologyPreservingLoss


# ============================================================
# Hugging Face Wrapper with Sliding-Window Validation
# ============================================================
class WrapperBasicUnet(nn.Module):
    """Wraps MONAI's BasicUnet to return HF-style outputs."""

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,  # Single channel for binary segmentation (more memory efficient)
        features: Sequence[int] = (32, 32, 64, 128, 256, 32),
        loss_config: dict | None = None,
        # *,
        # use_checkpoint: bool = True,
        # drop_rate: float = 0.0,
        # attn_drop_rate: float = 0.0,
    ) -> None:
        """Initialize model and loss.

        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
            features: Feature sizes for UNet encoder/decoder.
            loss_config: Dictionary with loss configuration (dice_weight, bce_weight, etc.).

        """
        super().__init__()
        self.model = BasicUnet(
            in_channels=in_channels,
            out_channels=out_channels,
            features=features,
        )

        # Initialize loss with config parameters if provided
        if loss_config is not None:
            self.loss_fn = ThinManifoldLoss(**loss_config)
        else:
            self.loss_fn = ThinManifoldLoss()

    def forward(
        self,
        pixel_values: torch.Tensor,
        label_fg_bin: torch.Tensor | None = None,
        label_valid: torch.Tensor | None = None,
        distance_map: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Run forward pass and optionally compute loss.

        Args:
            pixel_values: Input tensor [B, C, D, H, W].
            label_fg_bin: Binary foreground mask [B, 1, D, H, W], float32.
            label_valid: Valid region mask [B, 1, D, H, W], float32.
            distance_map: Distance transform of foreground [B, 1, D, H, W], float32.

        """
        logits = self.model(pixel_values)

        if label_fg_bin is not None and label_valid is not None:
            loss = self.loss_fn(logits, label_fg_bin, label_valid, distance_map)
            return {"loss": loss, "logits": logits}

        return {"logits": logits}

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        """Enable gradient checkpointing (stub for HF Trainer compatibility).

        BasicUnet doesn't support gradient checkpointing via this interface,
        but we provide the method to satisfy HuggingFace Trainer's requirements.
        """

    def gradient_checkpointing_disable(self):
        """Disable gradient checkpointing (stub for HF Trainer compatibility).

        BasicUnet doesn't support gradient checkpointing via this interface,
        but we provide the method to satisfy HuggingFace Trainer's requirements.
        """


# ============================================================
# Topology-Preserving UNet with Dual Heads
# ============================================================
class TopologyPreservingUNet(nn.Module):
    """UNet with dual output heads for topology-preserving segmentation.

    Architecture:
    - Shared encoder-decoder backbone (MONAI BasicUnet)
    - Binary segmentation head: outputs logits for binary classification
    - Distance regression head: outputs signed distance field (SDF) prediction

    The dual-head design leverages both:
    1. Voxel-wise classification (binary head with Dice/BCE/Tversky losses)
    2. Distance field regression (distance head with MSE + gradient penalty)

    Distance regression naturally enforces topological continuity because:
    - The SDF is smooth and continuous across the volume
    - Predicting distances requires understanding sheet geometry
    - Gradient penalty encourages smooth, connected predictions
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        base_channels: int = 32,
        loss_config: dict | None = None,
    ) -> None:
        """Initialize topology-preserving UNet.

        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels per head.
            features: Feature sizes for UNet encoder/decoder.
            loss_config: Dictionary with loss configuration including:
                - dice_weight, bce_weight, etc. (classification losses)
                - distance_weight: Weight for distance regression loss
                - gradient_penalty_weight: Weight for smoothness penalty

        """
        super().__init__()

        # -------------------------------
        # DynUNet configuration
        # -------------------------------
        self.backbone = DynUNet(
            spatial_dims=3,
            in_channels=in_channels,
            out_channels=base_channels,  # IMPORTANT: output features, not logits
            kernel_size=[3, 3, 3, 3],
            strides=[
                (1, 1, 1),  # preserve sheets
                (1, 2, 2),  # contextual XY pooling
                (2, 2, 2),
                (2, 2, 2),
            ],
            upsample_kernel_size=[
                (1, 2, 2),
                (2, 2, 2),
                (2, 2, 2),
            ],
            norm_name="instance",
            deep_supervision=False,
            res_block=True,  # VERY IMPORTANT for topology
        )
        feature_channels = base_channels

        # -------------------------------
        # Binary segmentation head
        # -------------------------------
        self.binary_head = nn.Sequential(
            nn.Conv3d(feature_channels, feature_channels // 2, kernel_size=3, padding=1),
            nn.InstanceNorm3d(feature_channels // 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(feature_channels // 2, out_channels, kernel_size=1),
        )

        # -------------------------------
        # Distance regression head (SDF)
        # -------------------------------
        self.distance_head = nn.Sequential(
            nn.Conv3d(feature_channels, feature_channels // 2, kernel_size=3, padding=1),
            nn.InstanceNorm3d(feature_channels // 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(feature_channels // 2, out_channels, kernel_size=1),
            nn.Tanh(),  # normalized SDF in [-1, 1]
        )

        # Initialize loss with config parameters
        if loss_config is not None:
            self.loss_fn = TopologyPreservingLoss(**loss_config)
        else:
            self.loss_fn = TopologyPreservingLoss()

    def forward(
        self,
        pixel_values: torch.Tensor,
        label_fg_bin: torch.Tensor | None = None,
        label_valid: torch.Tensor | None = None,
        distance_map: torch.Tensor | None = None,
        distance_unsigned: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Run forward pass through both heads.

        Args:
            pixel_values: Input tensor [B, C, D, H, W].
            label_fg_bin: Binary foreground mask [B, 1, D, H, W], float32.
            label_valid: Valid region mask [B, 1, D, H, W], float32.
            distance_map: Signed distance field (SDF) [B, 1, D, H, W], float32.
                         Used for surface loss (Kervadec-style).
            distance_unsigned: Unsigned distance from foreground [B, 1, D, H, W], float32.
                              Used as regression target for distance head.

        Returns:
            Dictionary with:
                - logits: Binary segmentation logits [B, 1, D, H, W]
                - distance_pred: Predicted SDF [B, 1, D, H, W]
                - loss: Combined loss (if labels provided)

        """
        # Shared encoder-decoder backbone
        features = self.backbone(pixel_values)

        # Dual heads
        logits = self.binary_head(features)
        distance_pred = self.distance_head(features)

        result = {
            "logits": logits,
            "distance_pred": distance_pred,
        }

        # Compute loss if training labels are provided
        if label_fg_bin is not None and label_valid is not None:
            loss = self.loss_fn(
                logits=logits,
                distance_pred=distance_pred,
                label_fg_bin=label_fg_bin,
                label_valid=label_valid,
                distance_map=distance_map,
                distance_unsigned=distance_unsigned,
            )
            result["loss"] = loss

        return result

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        """Enable gradient checkpointing (stub for HF Trainer compatibility)."""

    def gradient_checkpointing_disable(self):
        """Disable gradient checkpointing (stub for HF Trainer compatibility)."""
