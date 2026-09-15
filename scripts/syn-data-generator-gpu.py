"""Synthetic data generator for parallel arc sheets in 3D space.

Generates multiple parallel sheets with simple arc shapes along the z-axis.
Each sheet is a 2D manifold (surface) embedded in 3D space.

Usage:
    python syn-data-generator-gpu.py --num_cubes 10 --min_sheets 3 --max_sheets 8
"""

import argparse
import os
import shutil
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import pandas as pd
import torch
from scipy.ndimage import binary_dilation

from utils.util import metadata_summary


# ---------------------------------------------------------
# Generate parallel arc sheets along z-axis
# Simplified 2D manifolds in 3D space
# ---------------------------------------------------------
def generate_arc_sheet(
    y_center: float,
    arc_amplitude: float,
    arc_frequency: float,
    thickness: int,
    cube_size: int = 320,
    device: str = "cpu",
) -> np.ndarray:
    """Generate a single arc-shaped sheet parallel to the z-axis.

    The sheet is a 2D manifold (surface) in 3D space defined by:
    - x spans the entire cube (0 to cube_size)
    - y = y_center + amplitude * sin(frequency * x)  (the arc shape)
    - z spans the entire cube (0 to cube_size)

    Args:
        y_center: Center y-position of the sheet (0 to cube_size)
        arc_amplitude: Amplitude of the sinusoidal arc in voxels
        arc_frequency: Frequency of the arc (higher = more waves)
        thickness: Thickness of the sheet in voxels (5-10 typical)
        cube_size: Size of the cube (default 320)
        device: Device to use for computation

    Returns:
        3D numpy array mask of the sheet

    """
    mask = torch.zeros((cube_size, cube_size, cube_size), dtype=torch.uint8, device=device)

    # Create coordinate grids
    x_coords = torch.arange(cube_size, device=device, dtype=torch.float32)

    # For each x position, compute the y position of the arc center
    # y = y_center + amplitude * sin(frequency * 2π * x / cube_size)
    arc_y = y_center + arc_amplitude * torch.sin(arc_frequency * 2 * np.pi * x_coords / cube_size)

    # For each x, fill in the sheet with thickness centered on arc_y
    # The sheet spans all z values
    half_thickness = thickness / 2.0

    for x_idx in range(cube_size):
        y_arc_center = arc_y[x_idx].item()

        # Compute y range for this x position (with thickness)
        y_min = int(max(0, y_arc_center - half_thickness))
        y_max = int(min(cube_size, y_arc_center + half_thickness + 1))

        if y_min < y_max:
            # Fill all z values for this x and y range
            mask[x_idx, y_min:y_max, :] = 1

    return mask.cpu().numpy().astype(np.uint8)


def generate_parallel_arc_sheets(
    num_sheets: int,
    cube_size: int = 320,
    thickness_range: tuple = (5, 10),
    arc_amplitude_range: tuple = (10, 40),
    arc_frequency_range: tuple = (0.5, 2.0),
    device: str = "cpu",
) -> np.ndarray:
    """Generate multiple parallel arc sheets along the z-axis.

    Sheets are distributed along the y-axis with minimum spacing to avoid overlap.

    Args:
        num_sheets: Number of sheets to generate
        cube_size: Size of the cube (default 320)
        thickness_range: (min, max) thickness of sheets in voxels
        arc_amplitude_range: (min, max) amplitude of arc curves
        arc_frequency_range: (min, max) frequency of arc waves
        device: Device for computation

    Returns:
        Combined 3D mask with all sheets

    """
    final_mask = np.zeros((cube_size, cube_size, cube_size), dtype=np.uint8)

    # Distribute sheet centers evenly along y-axis with some randomness
    # Leave margin at edges for arc amplitude
    max_amplitude = arc_amplitude_range[1]
    margin = max_amplitude + thickness_range[1]

    available_range = cube_size - 2 * margin
    if num_sheets > 1:
        base_spacing = available_range / (num_sheets - 1)
        y_centers = [margin + i * base_spacing for i in range(num_sheets)]
        # Add small random perturbation (up to 20% of spacing)
        perturbation = base_spacing * 0.2
        y_centers = [yc + np.random.uniform(-perturbation, perturbation) for yc in y_centers]
    else:
        y_centers = [cube_size / 2]

    for y_center in y_centers:
        # Random parameters for this sheet
        thickness = np.random.randint(thickness_range[0], thickness_range[1] + 1)
        arc_amplitude = np.random.uniform(arc_amplitude_range[0], arc_amplitude_range[1])
        arc_frequency = np.random.uniform(arc_frequency_range[0], arc_frequency_range[1])

        sheet_mask = generate_arc_sheet(
            y_center=y_center,
            arc_amplitude=arc_amplitude,
            arc_frequency=arc_frequency,
            thickness=thickness,
            cube_size=cube_size,
            device=device,
        )

        # Combine with existing mask (union)
        final_mask = np.maximum(final_mask, sheet_mask)

    return final_mask


# ---------------------------------------------------------
# Convert mask → CT-like synthetic volume
# ---------------------------------------------------------
def create_ct_from_mask(mask, sheet_intensity=0.75, bg_intensity=0.15, device="cpu"):
    """Applies distance-based effects + noise for CT realism.

    Args:
        mask: Binary mask of sheet locations
        sheet_intensity: Intensity value for sheet voxels (0-1)
        bg_intensity: Intensity value for background voxels (0-1)
        device: Device for computation

    Returns:
        CT-like volume as uint8 numpy array (0-255)

    """
    mask_np = mask if isinstance(mask, np.ndarray) else mask.cpu().numpy()

    # Create boundary effect using dilation
    dilated = binary_dilation(mask_np, iterations=3).astype(np.float32)
    boundary_effect = dilated - mask_np.astype(np.float32)
    boundary_effect = np.clip(boundary_effect * 2, 0, 1)

    # Move to GPU for fast operations
    mask_tensor = torch.from_numpy(mask_np).float().to(device)
    boundary_tensor = torch.from_numpy(boundary_effect).float().to(device)

    # Create volume on GPU
    vol = torch.ones_like(mask_tensor, dtype=torch.float32, device=device) * bg_intensity
    vol += (sheet_intensity - bg_intensity) * mask_tensor
    # Add boundary highlight effect (brighter edges on sheets)
    vol += 0.10 * boundary_tensor

    # Add realistic noise on GPU
    vol += torch.normal(0, 0.03, size=mask_tensor.shape, device=device)
    # Slight intensity variation for realism
    vol *= torch.rand(mask_tensor.shape, device=device) * 0.08 + 0.96

    # Clamp to 0-1 and scale to 0-255
    vol = torch.clamp(vol, 0.0, 1.0) * 255.0
    return vol.cpu().numpy().astype(np.uint8)


# ---------------------------------------------------------
# Full synthetic cube generator
# ---------------------------------------------------------
def generate_synthetic_cube(
    cube_size=320,
    num_sheets=4,
    thickness_range=(5, 10),
    arc_amplitude_range=(10, 40),
    arc_frequency_range=(0.5, 2.0),
    device="cpu",
):
    """Generate synthetic cube with multiple parallel arc sheets.

    Args:
        cube_size: Size of the cube (default 320)
        num_sheets: Number of sheets to generate
        thickness_range: (min, max) thickness of sheets in voxels
        arc_amplitude_range: (min, max) amplitude of arc curves
        arc_frequency_range: (min, max) frequency of arc waves
        device: Device for computation

    Returns:
        Tuple of (ct_volume, mask) as numpy arrays

    """
    final_mask = generate_parallel_arc_sheets(
        num_sheets=num_sheets,
        cube_size=cube_size,
        thickness_range=thickness_range,
        arc_amplitude_range=arc_amplitude_range,
        arc_frequency_range=arc_frequency_range,
        device=device,
    )

    ct_volume = create_ct_from_mask(final_mask, device=device)
    return ct_volume, final_mask.astype(np.uint8)


# ---------------------------------------------------------
# Worker function for multi-GPU processing
# ---------------------------------------------------------
def generate_cube_worker(args):
    """Worker function for parallel cube generation."""
    (
        cube_idx,
        num_sheets,
        cube_size,
        thickness_range,
        arc_amplitude_range,
        arc_frequency_range,
        device,
    ) = args
    vol, mask = generate_synthetic_cube(
        cube_size=cube_size,
        num_sheets=num_sheets,
        thickness_range=thickness_range,
        arc_amplitude_range=arc_amplitude_range,
        arc_frequency_range=arc_frequency_range,
        device=device,
    )
    return cube_idx, vol, mask


# ---------------------------------------------------------
# Command-line interface
# ---------------------------------------------------------
def main():
    # Default save directory from .env
    nvme_dir = os.getenv("NVME_DIR", ".")
    if nvme_dir is None or nvme_dir == "":
        nvme_dir = "."
    print(f"[*] Using NVME_DIR: {nvme_dir}")

    out_dir = Path(nvme_dir) / "data"

    parser = argparse.ArgumentParser(
        description="Synthetic Vesuvius 3D dataset generator - Parallel Arc Sheets"
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=str(out_dir),
        help="Where to save .npy files",
    )
    parser.add_argument(
        "--out_folder",
        type=str,
        default="syn0",
        help="Output folder name",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--num_cubes", type=int, default=10)
    parser.add_argument(
        "--min_sheets",
        type=int,
        default=3,
        help="Minimum number of parallel sheets per cube",
    )
    parser.add_argument(
        "--max_sheets",
        type=int,
        default=8,
        help="Maximum number of parallel sheets per cube",
    )
    parser.add_argument("--cube_size", type=int, default=320)
    parser.add_argument(
        "--min_thickness",
        type=int,
        default=5,
        help="Minimum thickness of sheets in voxels",
    )
    parser.add_argument(
        "--max_thickness",
        type=int,
        default=10,
        help="Maximum thickness of sheets in voxels",
    )
    parser.add_argument(
        "--min_arc_amplitude",
        type=float,
        default=10.0,
        help="Minimum amplitude of arc curves in voxels",
    )
    parser.add_argument(
        "--max_arc_amplitude",
        type=float,
        default=40.0,
        help="Maximum amplitude of arc curves in voxels",
    )
    parser.add_argument(
        "--min_arc_frequency",
        type=float,
        default=0.5,
        help="Minimum frequency of arc waves",
    )
    parser.add_argument(
        "--max_arc_frequency",
        type=float,
        default=2.0,
        help="Maximum frequency of arc waves",
    )
    parser.add_argument(
        "--num_gpus",
        type=int,
        default=2,
        help="Number of GPUs to use",
    )
    args = parser.parse_args()

    # Set random seed for reproducibility
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Check GPU availability
    if not torch.cuda.is_available():
        print("[!] CUDA not available. Falling back to CPU.")
        devices = ["cpu"]
    else:
        num_available_gpus = torch.cuda.device_count()
        num_gpus = min(args.num_gpus, num_available_gpus)
        devices = [f"cuda:{i}" for i in range(num_gpus)]
        print(f"[+] Using {len(devices)} GPU(s): {devices}")

    # Output directory
    args.out_dir = Path(args.out_dir) / args.out_folder

    # Remove output directory if it exists
    if os.path.exists(args.out_dir):
        shutil.rmtree(args.out_dir)

    # Create directories
    train_images_save_dir = args.out_dir / "train_images"
    train_labels_save_dir = args.out_dir / "train_labels"
    os.makedirs(train_images_save_dir, exist_ok=True)
    os.makedirs(train_labels_save_dir, exist_ok=True)

    # Train data
    train_df = pd.DataFrame()

    # Prepare worker arguments
    worker_args = []
    for i in range(args.num_cubes):
        num_sheets = np.random.randint(args.min_sheets, args.max_sheets + 1)
        device = devices[i % len(devices)]  # Round-robin device assignment
        worker_args.append(
            (
                i,
                num_sheets,
                args.cube_size,
                (args.min_thickness, args.max_thickness),
                (args.min_arc_amplitude, args.max_arc_amplitude),
                (args.min_arc_frequency, args.max_arc_frequency),
                device,
            ),
        )

    # Use ThreadPool for GPU operations (better than ProcessPool for GPU)
    from multiprocessing.pool import ThreadPool

    # Increase workers based on GPU count
    max_workers = max(len(devices) * 8, 16)
    print(f"[+] Generating {args.num_cubes} cubes with {max_workers} workers...")

    # Track cube indices to maintain order for final CSV/metadata
    cube_metadata = []

    with ThreadPool(max_workers) as pool:
        for idx, (i, vol, mask) in enumerate(
            pool.imap_unordered(generate_cube_worker, worker_args),
        ):
            print(f"[+] Generated and saving cube {idx + 1}/{args.num_cubes} (ID: {i})")

            # Save immediately to disk to free memory
            id_name = 100_000 + i
            np.save(train_images_save_dir / f"{id_name}.npy", vol)
            np.save(train_labels_save_dir / f"{id_name}.npy", mask)

            # Track metadata for later
            cube_metadata.append((i, id_name))

            # Explicitly delete to free memory
            del vol, mask

    # Build train_df from metadata (cubes already on disk)
    for i, id_name in sorted(cube_metadata, key=lambda x: x[0]):
        train_df = pd.concat(
            [
                train_df,
                pd.DataFrame(
                    {
                        "id": [id_name],
                        "scroll_id": [1_000],
                    },
                ),
            ],
            ignore_index=True,
        )

    # Metadata summary
    meta_df = metadata_summary(
        train_images_save_dir,
        train_labels_save_dir,
        train_df,
        max_workers=16,
    )
    parquet_save_path = args.out_dir / "metadata_summary.parquet"
    meta_df.to_parquet(parquet_save_path)

    # Save train_df to disk
    train_df.to_csv(args.out_dir / "train.csv", index=False)
    print("[+] Done.")


if __name__ == "__main__":
    main()
