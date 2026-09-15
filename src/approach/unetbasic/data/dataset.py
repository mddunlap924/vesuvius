"""Dataset for 3D Vesuvius volume segmentation."""

import itertools
import json
import math
import os
import random
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from monai import transforms as MT
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
from scipy import ndimage
from scipy.ndimage import distance_transform_edt
from torch.utils.data import Dataset


# =============================================================================
# Custom MONAI Transforms for Label Preprocessing
# =============================================================================
class ExtractForeground(MT.MapTransform):
    """Extract foreground channel (label == 1) as float tensor.

    Converts 3-class labels (0=background, 1=foreground, 2=ignore) into
    binary foreground mask.
    """

    def __init__(self, keys, allow_missing_keys=False):
        super().__init__(keys, allow_missing_keys)

    def __call__(self, data):
        d = dict(data)
        for key in self.key_iterator(d):
            d[key + "_fg"] = (d[key] == 1).to(torch.float32)
        return d


class ExtractValidMask(MT.MapTransform):
    """Extract valid mask (label != 2) as float tensor.

    Creates a mask where 1.0 = valid region (background or foreground),
    0.0 = ignore region (padding).
    """

    def __init__(self, keys, allow_missing_keys=False):
        super().__init__(keys, allow_missing_keys)

    def __call__(self, data):
        d = dict(data)
        for key in self.key_iterator(d):
            d[key + "_valid"] = (d[key] != 2).to(torch.float32)
        return d


class CopyForeground(MT.MapTransform):
    """Copy binary foreground mask before distance transform.

    Creates a copy of label_fg as label_fg_bin to preserve binary ground truth
    while allowing label_fg to be transformed into a distance map.
    """

    def __init__(self, keys, allow_missing_keys=False):
        super().__init__(keys, allow_missing_keys)

    def __call__(self, data):
        d = dict(data)
        for key in self.key_iterator(d):
            # Copy the binary foreground mask
            d[key + "_bin"] = d[key].clone()
        return d


class BuildCropLabel(MT.MapTransform):
    """Build a cropping label that focuses on foreground boundaries.

    Produces `label_crop` (float in {0,1}) where 1 indicates boundary/foreground
    voxels of the sheet, excluding ignore regions (label==2).

    This is used as `label_key` for RandCropByPosNegLabeld so crops consistently
    contain manifold signal (critical for topology).
    """

    def __init__(
        self,
        keys,
        allow_missing_keys: bool = False,
        *,
        out_key: str = "label_crop",
        anisotropic: bool = True,
    ):
        super().__init__(keys, allow_missing_keys)
        self.out_key = out_key
        self.anisotropic = anisotropic

    def __call__(self, data):
        d = dict(data)
        for key in self.key_iterator(d):
            label = d[key]
            # Expect channel-first: [1, D, H, W]
            label_np = label.squeeze(0).cpu().numpy() if label.is_cuda else label.squeeze(0).numpy()

            valid = label_np != 2
            fg = (label_np == 1) & valid

            if fg.any():
                # Boundary emphasis: morphological gradient (dilation XOR erosion).
                # For thin 2D manifolds in 3D, prefer per-slice (1,3,3) structure.
                if self.anisotropic:
                    structure = np.ones((1, 3, 3), dtype=bool)
                else:
                    structure = np.ones((3, 3, 3), dtype=bool)
                dil = ndimage.binary_dilation(fg, structure=structure)
                ero = ndimage.binary_erosion(fg, structure=structure)
                boundary = np.logical_xor(dil, ero)
                crop_mask = np.logical_or(boundary, fg)
            else:
                crop_mask = np.zeros_like(label_np, dtype=bool)

            # Never sample inside ignore-only areas
            crop_mask &= valid

            d[self.out_key] = torch.as_tensor(crop_mask, dtype=torch.float32).unsqueeze(0)
        return d


class RandCropByPosNegLabeldSingle(MT.MapTransform):
    """RandCropByPosNegLabeld wrapper that always returns a single dict.

    MONAI's RandCropByPosNegLabeld returns a list of samples (len=num_samples).
    This wrapper keeps compatibility with the existing HF-style collator/trainer.
    """

    def __init__(
        self,
        keys,
        label_key: str,
        spatial_size,
        *,
        pos: float = 1.0,
        neg: float = 1.0,
        image_key: str | None = None,
        image_threshold: float = 0.0,
        allow_smaller: bool = False,
        allow_missing_keys: bool = False,
    ):
        super().__init__(keys, allow_missing_keys)
        self._crop = RandCropByPosNegLabeld(
            keys=keys,
            label_key=label_key,
            spatial_size=spatial_size,
            pos=pos,
            neg=neg,
            num_samples=1,
            image_key=image_key,
            image_threshold=image_threshold,
            allow_smaller=allow_smaller,
            allow_missing_keys=allow_missing_keys,
        )

    def __call__(self, data):
        out = self._crop(data)
        if isinstance(out, list):
            return out[0]
        return out


class ApplyUnsignedDistanceTransform(MT.MapTransform):
    """Apply unsigned distance transform to a binary mask.

    Computes the Euclidean distance from each voxel to the nearest foreground voxel.
    This is useful as a regression target for topology-preserving networks.

    Output:
        - 0 at foreground voxels
        - Positive distance for background voxels

    Note: This is a CPU bottleneck. For production, precompute distance maps
    offline and load them directly from disk.
    """

    def __init__(
        self,
        keys,
        allow_missing_keys: bool = False,
        *,
        normalize: bool = True,
        max_distance: float = 10.0,
    ):
        """Initialize the transform.

        Args:
            keys: Keys to apply transform to.
            allow_missing_keys: Whether to allow missing keys.
            normalize: Whether to normalize by max_distance.
            max_distance: Maximum distance for normalization/clamping.

        """
        super().__init__(keys, allow_missing_keys)
        self.normalize = normalize
        self.max_distance = max_distance

    def __call__(self, data):
        d = dict(data)
        for key in self.key_iterator(d):
            label_tensor = d[key]
            if label_tensor.is_cuda:
                label_np = label_tensor.squeeze(0).cpu().numpy()
            else:
                label_np = label_tensor.squeeze(0).numpy()

            # Unsigned distance transform from foreground
            fg = label_np > 0.5
            if fg.any():
                # Distance from each voxel to nearest foreground
                dist = distance_transform_edt(~fg)
            else:
                # No foreground - use max distance everywhere
                dist = np.full_like(label_np, self.max_distance, dtype=np.float32)

            # Clamp to max distance
            dist = np.clip(dist, 0, self.max_distance)

            if self.normalize:
                dist = dist / self.max_distance

            dist_tensor = torch.as_tensor(dist, dtype=label_tensor.dtype).unsqueeze(0)
            d[key + "_unsigned"] = dist_tensor

        return d


class ApplyDistanceTransform(MT.MapTransform):
    """Apply signed distance transform (SDF) to a binary mask.

    Uses asymmetric normalization to map SDF to [-1, 1] range, matching tanh output.
    This is important for thin sheets (~3 voxels) where inside distances are small
    but outside distances can be large.

    Normalization strategy:
    - Inside (negative) distances: normalized by max_inside_distance (default: 3.0)
    - Outside (positive) distances: normalized by max_outside_distance (default: 10.0)
    - Both are clipped and scaled to fill their respective half of [-1, 0] and [0, 1]

    This ensures the model's tanh output range is fully utilized.
    """

    def __init__(
        self,
        keys,
        allow_missing_keys: bool = False,
        *,
        normalize: bool = True,
        max_distance: float | None = None,
        max_inside_distance: float = 4.0,
        max_outside_distance: float = 10.0,
    ):
        """Initialize the distance transform.

        Args:
            keys: Keys to apply transform to.
            allow_missing_keys: Whether to allow missing keys.
            normalize: Whether to normalize to [-1, 1] range.
            max_distance: Legacy parameter for symmetric normalization.
                         If provided, overrides both inside/outside distances.
            max_inside_distance: Maximum inside (negative) distance for normalization.
                                Typical value: 3.0 for thin sheets (~3 voxels thick).
            max_outside_distance: Maximum outside (positive) distance for normalization.
                                 Typical value: 10.0 for background regions.

        """
        super().__init__(keys, allow_missing_keys)
        self.normalize = normalize
        self.max_distance = max_distance  # Legacy compatibility
        self.max_inside_distance = max_inside_distance
        self.max_outside_distance = max_outside_distance

    def __call__(self, data):
        d = dict(data)
        for key in self.key_iterator(d):
            label_tensor = d[key]
            if label_tensor.is_cuda:
                label_np = label_tensor.squeeze(0).cpu().numpy()
            else:
                label_np = label_tensor.squeeze(0).numpy()

            fg = label_np > 0.5
            if fg.any() and (~fg).any():
                dist_outside = distance_transform_edt(~fg)
                dist_inside = distance_transform_edt(fg)
                sdf = dist_outside - dist_inside
            else:
                sdf = np.zeros_like(label_np, dtype=np.float32)

            # Normalize to [-1, 1] range
            if self.normalize:
                if self.max_distance is not None and self.max_distance > 0:
                    # Legacy symmetric normalization for backward compatibility
                    sdf = np.clip(sdf, -self.max_distance, self.max_distance)
                    sdf = sdf / float(self.max_distance)
                else:
                    # Asymmetric normalization for thin sheets
                    # This maps the full SDF range to [-1, 1] to match tanh output
                    sdf_normalized = np.zeros_like(sdf, dtype=np.float32)

                    # Inside (negative): clip to [-max_inside, 0], scale to [-1, 0]
                    inside_mask = sdf < 0
                    if inside_mask.any():
                        inside_vals = np.clip(sdf[inside_mask], -self.max_inside_distance, 0)
                        sdf_normalized[inside_mask] = inside_vals / self.max_inside_distance

                    # Outside (positive): clip to [0, max_outside], scale to [0, 1]
                    outside_mask = sdf > 0
                    if outside_mask.any():
                        outside_vals = np.clip(sdf[outside_mask], 0, self.max_outside_distance)
                        sdf_normalized[outside_mask] = outside_vals / self.max_outside_distance

                    sdf = sdf_normalized
            # No normalization - just clip if distances are specified
            elif self.max_distance is not None and self.max_distance > 0:
                sdf = np.clip(sdf, -self.max_distance, self.max_distance)
            else:
                sdf = np.clip(sdf, -self.max_inside_distance, self.max_outside_distance)

            d[key] = torch.as_tensor(sdf, dtype=label_tensor.dtype).unsqueeze(0)

        return d


class RandCoarseDropoutdWithIgnore(MT.MapTransform):
    """Apply RandCoarseDropoutd with different fill values for image and label.

    Best practice for cutout augmentation:
    - Fills image cutout regions with random noise matching image statistics
    - Sets label to ignore value (2) in same regions
    - Loss function will skip these regions via label_valid mask
    """

    def __init__(
        self,
        keys,
        prob=0.1,
        holes=1,
        spatial_size=(10, 10, 10),
        fill_value_label=2,
        max_holes=None,
        max_spatial_size=None,
        allow_missing_keys=False,
        use_noise=True,
    ):
        super().__init__(keys, allow_missing_keys)
        self.prob = prob
        self.holes = holes
        self.spatial_size = spatial_size
        self.fill_value_label = fill_value_label
        self.max_holes = max_holes
        self.max_spatial_size = max_spatial_size
        self.use_noise = use_noise

        # Create dropout transform for label only
        # Image will be handled with custom noise filling
        self.dropout_label = RandCoarseDropoutd(
            keys=["label"],
            prob=prob,
            holes=holes,
            spatial_size=spatial_size,
            fill_value=fill_value_label,
            max_holes=max_holes,
            max_spatial_size=max_spatial_size,
        )

    def __call__(self, data):
        d = dict(data)

        # Apply random check for probability
        if random.random() > self.prob:
            return d

        # Get image tensor
        image = d["image"]
        label = d["label"]

        # Store random state for synchronized dropout
        rand_state = random.getstate()
        np_rand_state = np.random.get_state()

        # Apply label dropout to get the mask locations
        d_temp = {"label": label}
        d_temp = self.dropout_label(d_temp)
        modified_label = d_temp["label"]

        # Create mask of dropout regions (where label changed to fill_value_label)
        dropout_mask = label != modified_label

        # Fill image dropout regions with noise
        if self.use_noise and dropout_mask.any():
            # Calculate image statistics for noise generation
            mu = image.mean()
            sigma = image.std()

            # Generate noise matching image statistics
            noise = torch.randn_like(image) * sigma + mu

            # Apply noise only to dropout regions
            image = torch.where(dropout_mask, noise, image)
        else:
            # Fallback to zeros if noise disabled
            image = torch.where(dropout_mask, torch.zeros_like(image), image)

        # Update data dict
        d["image"] = image
        d["label"] = modified_label

        return d


# Add this custom transform class before PapyrusDataset
class CenterCropd:
    """Center crop to patch_size, handling variable input sizes (256, 320, 384)."""

    def __init__(self, keys, patch_size):
        self.keys = keys
        self.patch_size = patch_size

    def __call__(self, data):
        for key in self.keys:
            img = data[key]
            # Get actual shape (assuming channel-first after EnsureChannelFirstd)
            shape = img.shape[1:]  # Skip channel dimension
            # Compute center crop start position
            roi_start = tuple(max(0, (s - p) // 2) for s, p in zip(shape, self.patch_size))
            # Crop
            slices = tuple(
                slice(start, start + patch_size)
                for start, patch_size in zip(roi_start, self.patch_size)
            )
            data[key] = img[(slice(None),) + slices]  # Keep channel dim
        return data


def build_augmentation_transforms(
    augmentation_config: Any | None,
    *,
    keys: tuple[str, str] = ("image", "label"),
) -> list[Any]:
    """Build MONAI dict transforms from an OmegaConf/dict config.

    Expected YAML shape:
        augmentation:
          RandFlipd:
            apply: true
            prob: 0.5
            spatial_axis: [1, 2]
          RandRotate90d:
            apply: true
            prob: 0.5
            max_k: 3

    Only transforms explicitly supported here will be instantiated.
    """
    if augmentation_config is None:
        return []

    if isinstance(augmentation_config, DictConfig):
        augmentation_config = OmegaConf.to_container(augmentation_config, resolve=True)

    if not isinstance(augmentation_config, dict):
        msg = f"augmentation_config must be a mapping, got {type(augmentation_config)!r}"
        raise TypeError(msg)

    supported = {
        "Rand3DElasticd": Rand3DElasticd,
        "RandAdjustContrastd": RandAdjustContrastd,
        "RandCoarseDropoutd": RandCoarseDropoutd,
        "RandCoarseDropoutdWithIgnore": RandCoarseDropoutdWithIgnore,
        "RandFlipd": RandFlipd,
        "RandGaussianNoised": RandGaussianNoised,
        "RandGaussianSmoothd": RandGaussianSmoothd,
        "RandRotate90d": RandRotate90d,
        "RandScaleIntensityd": RandScaleIntensityd,
        "RandShiftIntensityd": RandShiftIntensityd,
        "RandZoomd": RandZoomd,
    }

    transforms: list[Any] = []
    for name, params in augmentation_config.items():
        if name not in supported:
            msg = (
                f"Unsupported augmentation transform: {name}. Supported: {sorted(supported.keys())}"
            )
            raise ValueError(msg)

        if params is None:
            params = {}
        if isinstance(params, DictConfig):
            params = OmegaConf.to_container(params, resolve=True)
        if not isinstance(params, dict):
            msg = f"Augmentation params for {name} must be a mapping, got {type(params)!r}"
            raise TypeError(msg)

        if not params.get("apply", True):
            continue

        init_kwargs = {k: v for k, v in params.items() if k not in ("apply", "keys_applied")}
        # Use keys_applied from config if provided, otherwise use all keys
        transform_keys = params.get("keys_applied", list(keys))
        init_kwargs.setdefault("keys", transform_keys)

        # Special handling for RandCoarseDropoutdWithIgnore
        if name == "RandCoarseDropoutdWithIgnore":
            # Remove keys from kwargs as it's passed separately
            init_kwargs.pop("keys", None)
            transforms.append(supported[name](keys=transform_keys, **init_kwargs))
        else:
            transforms.append(supported[name](**init_kwargs))

    return transforms


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

        if self.augmentation:
            # Training transforms with augmentations
            # IMPORTANT: Spatial augmentations (flip, rotate, cutout) must occur BEFORE
            # label extraction to ensure cutout regions are properly marked as ignore (label=2)
            base_transforms = [
                LoadImaged(keys=["image", "label"]),
                EnsureChannelFirstd(keys=["image", "label"], channel_dim="no_channel"),
                ScaleIntensityRanged(
                    keys=["image"],
                    a_min=0,
                    a_max=255,
                    b_min=0.0,
                    b_max=1.0,
                    clip=True,
                ),
                SpatialPadd(
                    keys=["image"],
                    spatial_size=self.patch_size,
                    mode="constant",
                    value=0,  # image padding -> zero intensity
                ),
                SpatialPadd(
                    keys=["label"],
                    spatial_size=self.patch_size,
                    mode="constant",
                    value=2,  # label padding -> mask class
                ),
                # Foreground/boundary-aware cropping for thin manifolds.
                # Build a binary crop mask that emphasizes label==1 boundaries (excluding ignore==2),
                # then sample a crop around positives.
                BuildCropLabel(keys=["label"], out_key="label_crop", anisotropic=True),
                RandCropByPosNegLabeldSingle(
                    keys=["image", "label"],
                    label_key="label_crop",
                    spatial_size=self.patch_size,
                    pos=1.0,  # More balanced sampling across Z-axis
                    neg=0.0,  # Equal weight to background regions
                    image_key="image",
                    image_threshold=0.0,
                    allow_smaller=False,
                ),
            ]

            # Spatial augmentations applied BEFORE label extraction
            # This ensures cutout regions get marked as ignore (label=2)
            aug_transforms = (
                build_augmentation_transforms(augmentation_config, keys=("image", "label"))
                if augmentation
                else []
            )

            # Label extraction happens AFTER augmentations
            # This way cutout regions (label=2) are properly marked as invalid
            label_transforms = [
                # Extract foreground and valid masks from 3-class labels
                ExtractForeground(keys=["label"]),
                # Copy binary foreground before distance transform
                CopyForeground(keys=["label_fg"]),
                ExtractValidMask(keys=["label"]),
                # Apply signed distance transform to label_fg (in-place) -> label_fg becomes SDF
                # Asymmetric normalization: inside max=3 (thin sheets), outside max=10
                # Maps SDF to [-1, 1] to match tanh output from distance head
                ApplyDistanceTransform(keys=["label_fg"], normalize=True),
                # Apply unsigned distance transform to label_fg_bin -> creates label_fg_bin_unsigned
                ApplyUnsignedDistanceTransform(
                    keys=["label_fg_bin"],
                    normalize=True,
                    max_distance=10.0,
                ),
            ]

            self.transforms = Compose([*base_transforms, *aug_transforms, *label_transforms])
        else:
            # Val/test transforms (no augmentations)
            self.transforms = Compose(
                [
                    LoadImaged(keys=["image", "label"]),
                    EnsureChannelFirstd(keys=["image", "label"]),
                    ScaleIntensityRanged(
                        keys=["image"],
                        a_min=0,
                        a_max=255,
                        b_min=0.0,
                        b_max=1.0,
                        clip=True,
                    ),
                    # Extract foreground and valid masks from 3-class labels
                    ExtractForeground(keys=["label"]),
                    # Copy binary foreground before distance transform
                    CopyForeground(keys=["label_fg"]),
                    ExtractValidMask(keys=["label"]),
                    # Apply signed distance transform to label_fg (in-place) -> label_fg becomes SDF
                    # Asymmetric normalization: inside max=3 (thin sheets), outside max=10
                    # Maps SDF to [-1, 1] to match tanh output from distance head
                    ApplyDistanceTransform(keys=["label_fg"], normalize=True),
                    # Apply unsigned distance transform to label_fg_bin -> creates label_fg_bin_unsigned
                    ApplyUnsignedDistanceTransform(
                        keys=["label_fg_bin"],
                        normalize=True,
                        max_distance=10.0,
                    ),
                ],
            )

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        image_id = self.image_ids[idx]
        d = self.transforms(
            {
                "image": self.data_dir / f"train_images/{image_id}.npy",
                "label": self.data_dir / f"train_labels/{image_id}.npy",
            },
        )
        # Use as_tensor to avoid memory copy when possible (shares memory with numpy)
        # Keep as float32 - autocast will handle fp16 conversion during training
        # Note: Tensors are on CPU here; pin_memory in DataLoader handles GPU transfer
        pixel_values = torch.as_tensor(d["image"], dtype=torch.float32)
        label_fg_bin = torch.as_tensor(d["label_fg_bin"], dtype=torch.float32)
        label_fg = torch.as_tensor(
            d["label_fg"],
            dtype=torch.float32,
        )  # signed distance field (SDF)
        label_valid = torch.as_tensor(d["label_valid"], dtype=torch.float32)
        # Unsigned distance from foreground (normalized, 0 at foreground, >0 outside)
        distance_unsigned = torch.as_tensor(d["label_fg_bin_unsigned"], dtype=torch.float32)

        return {
            # HF model wrapper expects `pixel_values` in forward signature
            "pixel_values": pixel_values,
            "label_fg_bin": label_fg_bin,  # binary ground truth [0, 1]
            "label_fg": label_fg,  # signed distance field (SDF), used for surface loss
            "label_valid": label_valid,  # valid region mask
            "distance_unsigned": distance_unsigned,  # unsigned distance from foreground (regression target)
            "idx": idx,
            "id": image_id,  # Use actual image ID if available
        }
