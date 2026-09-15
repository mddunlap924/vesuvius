#!/usr/bin/env python
"""Wrapper for nnUNetv2_predict that limits VRAM usage.

This wrapper sets memory constraints to match Kaggle T4 behaviour (~4GB on a 16GB T4-class GPU
versus ~19GB on a 24GB desktop GPU).

The problem: PyTorch 2.9.1's conv3d with default memory format (NCDHW) uses extremely
memory-inefficient algorithms, trying to allocate 75GB+ for operations that should use ~6GB.

Solution: Convert the network and inputs to channels_last_3d format (NDHWC) which uses
much more efficient CUDA kernels.

This wrapper monkey-patches nnUNet to use channels_last_3d format.
"""

import os
import sys

# ============================================================================
# CRITICAL: Set these BEFORE importing torch!
# ============================================================================

# Memory allocation config
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True,garbage_collection_threshold:0.6"

# Now import torch
import torch

# ============================================================================
# Set cuDNN options
# ============================================================================
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = False  # Disable benchmarking
torch.backends.cudnn.deterministic = True  # Use deterministic algorithms
torch.backends.cudnn.allow_tf32 = True  # Allow TF32 for speed

# Print settings for debugging
print(f"[WRAPPER] PyTorch version: {torch.__version__}")
print(f"[WRAPPER] cuDNN version: {torch.backends.cudnn.version()}")
print(f"[WRAPPER] cuDNN benchmark: {torch.backends.cudnn.benchmark}")
print("[WRAPPER] Using channels_last_3d memory format for memory efficiency")

# ============================================================================
# Monkey-patch nnUNet to use channels_last_3d format
# ============================================================================

import nnunetv2.inference.predict_from_raw_data as nnunet_predict

# Store original __init__
_original_init = nnunet_predict.nnUNetPredictor.__init__


def _patched_init(self, *args, **kwargs):
    """Patched __init__ that configures memory-efficient settings."""
    # Let original init run
    _original_init(self, *args, **kwargs)

    # Override cuDNN settings
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    print(f"[WRAPPER] After init - cuDNN benchmark: {torch.backends.cudnn.benchmark}")


# Apply init patch
nnunet_predict.nnUNetPredictor.__init__ = _patched_init

# ============================================================================
# Patch the network loading to convert to channels_last_3d
# ============================================================================

_original_initialize_from_trained = (
    nnunet_predict.nnUNetPredictor.initialize_from_trained_model_folder
)


def _patched_initialize_from_trained(self, *args, **kwargs):
    """Patch to convert network to channels_last_3d after loading."""
    # Call original
    result = _original_initialize_from_trained(self, *args, **kwargs)

    # Convert network to channels_last_3d format for memory efficiency
    if hasattr(self, "network") and self.network is not None:
        print("[WRAPPER] Converting network to channels_last_3d format...")
        self.network = self.network.to(memory_format=torch.channels_last_3d)
        print("[WRAPPER] Network converted to channels_last_3d")

    return result


nnunet_predict.nnUNetPredictor.initialize_from_trained_model_folder = (
    _patched_initialize_from_trained
)

# ============================================================================
# Patch _internal_maybe_mirror_and_predict to ensure input is channels_last_3d
# ============================================================================

_original_maybe_mirror = nnunet_predict.nnUNetPredictor._internal_maybe_mirror_and_predict


def _patched_maybe_mirror(self, x):
    """Ensure input tensor uses channels_last_3d format."""
    # Convert input to channels_last_3d
    if x.dim() == 5:  # 3D data: (N, C, D, H, W)
        x = x.to(memory_format=torch.channels_last_3d)
    return _original_maybe_mirror(self, x)


nnunet_predict.nnUNetPredictor._internal_maybe_mirror_and_predict = _patched_maybe_mirror

# ============================================================================
# Run nnUNetv2_predict
# ============================================================================

from nnunetv2.inference.predict_from_raw_data import predict_entry_point

if __name__ == "__main__":
    predict_entry_point()
