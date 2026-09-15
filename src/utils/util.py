from concurrent.futures import ThreadPoolExecutor, as_completed
from multiprocessing import cpu_count
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile as tiff


def image_metadata(image: np.ndarray, kind: str = "image") -> pd.Series:
    """Return metadata for a numpy image or mask array as a pandas.Series.
    kind: 'image' (default) or 'mask'. If 'mask', also return percentages of values 0,1,2.
    Shape components x,y,z are provided where possible (x = last dim, y = second-last, z = third-last).
    """
    arr = np.asarray(image)
    stats = {}
    stats["kind"] = str(kind)
    stats["shape"] = tuple(arr.shape)
    stats["ndim"] = int(arr.ndim)
    stats["dtype"] = str(arr.dtype)
    stats["size"] = int(arr.size)
    stats["n_bytes"] = int(arr.nbytes)

    # Assign x,y,z using last three dimensions when available
    if arr.ndim >= 1:
        stats["x"] = int(arr.shape[-1])
    else:
        stats["x"] = None
    if arr.ndim >= 2:
        stats["y"] = int(arr.shape[-2])
    else:
        stats["y"] = None
    if arr.ndim >= 3:
        stats["z"] = int(arr.shape[-3])
    else:
        stats["z"] = None

    # NaN / Inf counts (only meaningful for floating types)
    if np.issubdtype(arr.dtype, np.floating):
        stats["n_nan"] = int(np.isnan(arr).sum())
        stats["n_inf"] = int(np.isinf(arr).sum())
    else:
        stats["n_nan"] = 0
        stats["n_inf"] = 0

    # Unique values (may be expensive for very large arrays)
    try:
        stats["n_unique"] = int(np.unique(arr).size)
    except Exception:
        stats["n_unique"] = None

    # Numeric summaries when applicable
    if np.issubdtype(arr.dtype, np.number):
        flat = arr.ravel()
        if np.issubdtype(arr.dtype, np.floating):
            valid = flat[np.isfinite(flat)]
        else:
            valid = flat

        if valid.size > 0:
            stats["min"] = float(np.min(valid))
            stats["max"] = float(np.max(valid))
            stats["mean"] = float(np.mean(valid))
            stats["std"] = float(np.std(valid))
            stats["median"] = float(np.median(valid))
            stats["percentile_25"] = float(np.percentile(valid, 25))
            stats["percentile_75"] = float(np.percentile(valid, 75))
            stats["zeros"] = int((valid == 0).sum())
        else:
            for k in [
                "min",
                "max",
                "mean",
                "std",
                "median",
                "percentile_25",
                "percentile_75",
                "zeros",
            ]:
                stats[k] = None
    else:
        for k in ["min", "max", "mean", "std", "median", "percentile_25", "percentile_75", "zeros"]:
            stats[k] = None

    # Mask-specific summaries: percentages of values 0,1,2 (as 0-100 floats)
    if str(kind).lower() == "mask":
        flat = arr.ravel()
        total = max(0, flat.size)
        for v in (0, 1, 2):
            cnt = int((flat == v).sum())
            stats[f"count_{v}"] = cnt
            stats[f"percent_{v}"] = (100.0 * cnt / total) if total > 0 else None
    else:
        # For non-mask arrays ensure consistent keys count_0/1/2 and percent_0/1/2
        flat = arr.ravel()
        total = max(0, flat.size)
        # count_0 derived from zeros; other counts not applicable for images
        stats["count_0"] = int(stats.get("zeros", 0) if stats.get("zeros") is not None else 0)
        stats["percent_0"] = (100.0 * stats["count_0"] / total) if total > 0 else None
        for v in (1, 2):
            stats[f"count_{v}"] = None
            stats[f"percent_{v}"] = None

    return pd.Series(stats)


def metadata_summary(
    train_images_dir,
    train_labels_dir,
    train_df: pd.DataFrame,
    *,
    max_workers=None,
) -> pd.DataFrame:
    """Generate metadata summary for all images and masks in LOCAL_DATA_DIR."""
    # Gather all unique ids from train_df (assumes train_df was loaded earlier)
    ids = train_df["id"].unique().tolist()

    # Map id -> scroll_id for quick lookup
    id_to_scroll = train_df.set_index("id")["scroll_id"].to_dict()

    rows = []

    def _process_id(img_id):
        """Load image and mask, compute metadata, return a flat row dict."""
        img_path = train_images_dir / f"{img_id}.npy"
        mask_path = train_labels_dir / f"{img_id}.npy"
        try:
            img = np.load(img_path)
        except Exception as e:
            return {
                "id": str(img_id),
                "scroll_id": id_to_scroll.get(img_id),
                "error": f"read_image_error: {e}",
            }
        try:
            mask = np.load(mask_path)
        except Exception:
            mask = None

        img_meta = image_metadata(img, kind="image").to_dict()
        # build mask_meta explicitly so 'mask_kind' is recorded even when the mask file is missing
        if mask is not None:
            mask_meta = image_metadata(mask, kind="mask").to_dict()
        else:
            # default keys based on img_meta keys but set kind to 'mask' and ensure consistent count/percent keys
            mask_meta = dict.fromkeys(img_meta.keys())
            mask_meta["kind"] = "mask"
            mask_meta.setdefault("count_0", None)
            mask_meta.setdefault("percent_0", None)
            mask_meta.setdefault("count_1", None)
            mask_meta.setdefault("percent_1", None)
            mask_meta.setdefault("count_2", None)
            mask_meta.setdefault("percent_2", None)

        row = {"id": str(img_id), "scroll_id": id_to_scroll.get(img_id, None)}
        row.update({f"img_{k}": v for k, v in img_meta.items()})
        row.update({f"mask_{k}": v for k, v in mask_meta.items()})
        return row

    # Configure worker count (tune as needed)
    if max_workers is None:
        max_workers = min(2, (cpu_count() or 1) * 4)

    print(f"Using max_workers={max_workers}")

    rows = []
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures_map = {ex.submit(_process_id, img_id): img_id for img_id in ids}
        for i, fut in enumerate(as_completed(futures_map), start=1):
            img_id = futures_map[fut]
            try:
                row = fut.result()
                rows.append(row)
            except Exception as e:
                rows.append(
                    {
                        "id": str(img_id),
                        "scroll_id": id_to_scroll.get(img_id),
                        "error": f"processing_error: {e}",
                    },
                )
            if i % 100 == 0:
                print(f"Processed {i}/{len(ids)}")

    df_meta = pd.DataFrame(rows)
    return df_meta
