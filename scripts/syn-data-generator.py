"""uv run generate_synthetic.py \
    --num_cubes 50 \
    --min_sheets 1 \
    --max_sheets 10 \
    --min_thickness 1 \
    --max_thickness 3 \
    --fray_strength 1.2
"""

import argparse
import os
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.interpolate import splev, splprep
from scipy.ndimage import distance_transform_edt, gaussian_filter

from utils.util import metadata_summary


# ---------------------------------------------------------
# Utility: random smooth 3D curve for driving sheet geometry
# ---------------------------------------------------------
def generate_random_spline(num_points=5, cube_size=320, scale=0.8):
    """Generate a smooth 3D spline curve used to construct curved sheets."""
    pts = (np.random.rand(3, num_points) - 0.5) * scale * cube_size
    tck, _ = splprep(pts, s=5)
    u_fine = np.linspace(0, 1, 200)
    x, y, z = splev(u_fine, tck)
    return np.vstack([x, y, z]).T  # shape (200, 3)


# ---------------------------------------------------------
# Create a thin 3D sheet around a spline by extruding normal vectors
# ---------------------------------------------------------
def create_sheet_from_spline(curve, thickness=2, width=40, cube_size=320):
    """Extrudes a spline into a thin, curved 2D manifold embedded in 3D."""
    mask = np.zeros((cube_size, cube_size, cube_size), dtype=np.uint8)

    for i in range(1, len(curve) - 1):
        p_prev = curve[i - 1]
        p = curve[i]
        p_next = curve[i + 1]

        # Tangent
        t = p_next - p_prev
        t /= np.linalg.norm(t) + 1e-6

        # Random orthogonal normal vectors
        n1 = np.cross(t, np.random.randn(3))
        n1 /= np.linalg.norm(n1) + 1e-6
        n2 = np.cross(t, n1)
        n2 /= np.linalg.norm(n2) + 1e-6

        # Local cross-section grid
        for w in np.linspace(-width, width, 80):
            for th in np.linspace(-thickness / 2, thickness / 2, thickness):
                point = p + w * n1 + th * n2
                x, y, z = np.round(point).astype(int)
                if 0 <= x < cube_size and 0 <= y < cube_size and 0 <= z < cube_size:
                    mask[x, y, z] = 1

    return mask


# ---------------------------------------------------------
# Add fraying / irregularity to sheet
# ---------------------------------------------------------
def fray_mask(mask, fray_strength=1.5):
    """Erode/dilate with noise to create rough sheet edges."""
    noise = gaussian_filter(np.random.randn(*mask.shape), sigma=3)
    noisy = mask.astype(float) + fray_strength * (noise > 0.7)
    return (noisy > 0.5).astype(np.uint8)


# ---------------------------------------------------------
# Convert mask → CT-like synthetic volume
# ---------------------------------------------------------
def create_ct_from_mask(mask, sheet_intensity=0.55, bg_intensity=0.35):
    """Applies distance-based effects + noise for CT realism."""
    vol = np.ones_like(mask, dtype=np.float32) * bg_intensity

    # Distance transform for soft boundary gradients
    dist = distance_transform_edt(mask == 0)
    boundary_effect = np.exp(-dist / 3.0)

    vol += (sheet_intensity - bg_intensity) * (mask.astype(float))
    vol += 0.20 * boundary_effect * mask

    # Add CT-like Gaussian + speckle noise
    vol += np.random.normal(0, 0.02, mask.shape)
    vol *= np.random.uniform(0.95, 1.05, size=mask.shape)  # speckle

    return np.clip(vol, 0.0, 1.0)


# ---------------------------------------------------------
# Full synthetic cube generator
# ---------------------------------------------------------
def generate_synthetic_cube(
    cube_size=320,
    num_sheets=4,  # now passed in randomly
    thickness_range=(1, 3),
    width_range=(30, 60),
    fray_strength=1.0,
):
    final_mask = np.zeros((cube_size, cube_size, cube_size), dtype=np.uint8)

    for _ in range(num_sheets):
        curve = generate_random_spline()
        thickness = np.random.randint(thickness_range[0], thickness_range[1] + 1)
        width = np.random.randint(width_range[0], width_range[1] + 1)

        sheet_mask = create_sheet_from_spline(curve, thickness, width, cube_size)
        sheet_mask = fray_mask(sheet_mask, fray_strength)

        final_mask = np.maximum(final_mask, sheet_mask)

    ct_volume = create_ct_from_mask(final_mask)
    return ct_volume.astype(np.float32), final_mask.astype(np.uint8)


# ---------------------------------------------------------
# Command-line interface
# ---------------------------------------------------------
def main():
    # Default save directory from .env
    nvme_dir = os.getenv("NVME_DIR", ".")
    if nvme_dir is None or nvme_dir == "":
        nvme_dir = "."
    out_dir = Path(nvme_dir) / "data"

    parser = argparse.ArgumentParser(description="Synthetic Vesuvius 3D dataset generator")
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
    parser.add_argument("--min_sheets", type=int, default=1)
    parser.add_argument("--max_sheets", type=int, default=10)
    parser.add_argument("--cube_size", type=int, default=320)
    parser.add_argument("--min_thickness", type=int, default=1)
    parser.add_argument("--max_thickness", type=int, default=3)
    parser.add_argument("--fray_strength", type=float, default=1.0)
    parser.add_argument("--min_width", type=int, default=30)
    parser.add_argument("--max_width", type=int, default=60)
    args = parser.parse_args()

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

    for i in range(args.num_cubes):
        print(f"[+] Generating cube {i + 1}/{args.num_cubes} ...")

        # Random number of sheets for this cube
        num_sheets = np.random.randint(args.min_sheets, args.max_sheets + 1)

        vol, mask = generate_synthetic_cube(
            cube_size=args.cube_size,
            num_sheets=num_sheets,
            thickness_range=(args.min_thickness, args.max_thickness),
            width_range=(args.min_width, args.max_width),
            fray_strength=args.fray_strength,
        )
        id_name = 100_000 + i
        np.save(train_images_save_dir / f"{id_name}.npy", vol)
        np.save(train_labels_save_dir / f"{id_name}.npy", mask)
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
    print("Done.")


if __name__ == "__main__":
    main()
