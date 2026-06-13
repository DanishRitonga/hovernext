"""PanNuke dataset loading — converts HuggingFace PanNuke to CONIC format for HoVer-NeXt training.

The upstream HoVer-NeXt `src/data_utils.py:get_pannuke()` expects data laid out as:

    data/pannuke_conic/
      images/fold1/images.npy   (N, 256, 256, 3) uint8
      images/fold1/types.npy    (N,)             <U30 tissue-type strings
      images/fold2/...
      images/fold3/...
      masks/fold1/labels.npy    (N, 256, 256, 2) int32  [instance_id, class_id]
      masks/fold2/...
      masks/fold3/...
"""

from __future__ import annotations

import gc
from pathlib import Path

import numpy as np
from datasets import load_dataset
from PIL import Image
from tqdm import tqdm

DATASET_ID = "RationAI/PanNuke"
DATA_DIR = Path("data") / "pannuke_conic"

CELL_TYPES = ["Neoplastic", "Inflammatory", "Connective", "Dead", "Epithelial"]

FOLD_NAMES = ["fold1", "fold2", "fold3"]


def _instances_to_labelmap(instances: list, img_h: int, img_w: int) -> np.ndarray:
    """Convert a list of binary instance masks into a single integer label map.

    Returns:
        (img_h, img_w) int32 array where 0=background, 1,2,...=instance IDs.
    """
    labelmap = np.zeros((img_h, img_w), dtype=np.int32)
    for idx, mask in enumerate(instances, start=1):
        m = np.array(mask)
        if m.ndim == 3:
            m = m[..., 0]
        m = (m > 0).astype(bool)
        labelmap[m] = idx
    return labelmap


def _build_class_map(
    labelmap: np.ndarray,
    categories: list[int],
) -> np.ndarray:
    """Build a per-pixel class map from instance label map and per-instance categories.

    Args:
        labelmap: (H, W) int32 instance label map (0=bg, 1,2,...=instances).
        categories: List of cell-type indices (0-indexed), one per instance,
                    in the same order as instance IDs (1, 2, ...).

    Returns:
        (H, W) int32 class map where 0=background, 1..5=cell types
        (PanNuke categories are 0-indexed, so we add 1 to shift to 1-indexed).
    """
    class_map = np.zeros_like(labelmap)
    for inst_id in range(1, labelmap.max() + 1):
        cat_idx = inst_id - 1
        if cat_idx < len(categories):
            class_map[labelmap == inst_id] = int(categories[cat_idx]) + 1
    return class_map


def _fold_has_data(fold_name: str) -> bool:
    """Check whether a fold's CONIC arrays already exist on disk."""
    img_path = DATA_DIR / "images" / fold_name / "images.npy"
    lbl_path = DATA_DIR / "masks" / fold_name / "labels.npy"
    return img_path.exists() and lbl_path.exists()


def _process_fold(
    fold_name: str,
    max_samples: int | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Stream one PanNuke fold from HuggingFace and return stacked arrays.

    Returns:
        images: (N, H, W, 3) uint8
        labels: (N, H, W, 2) int32  [instance, class]
        types:  (N,) <U30 tissue-type strings
    """
    ds = load_dataset(DATASET_ID, split=fold_name, streaming=True)
    if max_samples is not None:
        ds = ds.take(max_samples)

    images: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    types: list[str] = []

    i = -1
    for i, sample in enumerate(tqdm(ds, desc=fold_name, unit="img")):
        img: Image.Image = sample["image"]
        img_w, img_h = img.size
        img_np = np.array(img, dtype=np.uint8)

        labelmap = _instances_to_labelmap(sample["instances"], img_h, img_w)
        class_map = _build_class_map(labelmap, sample["categories"])

        label_2ch = np.stack([labelmap, class_map], axis=-1).astype(np.int32)

        images.append(img_np)
        labels.append(label_2ch)
        types.append(sample.get("tissue", "Unknown"))

        if i % 100 == 0:
            gc.collect()

    if i < 0:
        raise RuntimeError(f"No samples loaded from {fold_name}")

    images_arr = np.stack(images, axis=0)
    labels_arr = np.stack(labels, axis=0)
    types_arr = np.array(types, dtype="<U30")

    n_instances = int(labels_arr[..., 0].max())
    print(f"  {fold_name}: {i+1} images, {n_instances} instances")

    return images_arr, labels_arr, types_arr


def _save_fold(fold_name: str, images: np.ndarray, labels: np.ndarray, types: np.ndarray):
    img_dir = DATA_DIR / "images" / fold_name
    mask_dir = DATA_DIR / "masks" / fold_name
    img_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)

    np.save(img_dir / "images.npy", images)
    np.save(img_dir / "types.npy", types)
    np.save(mask_dir / "labels.npy", labels)
    print(f"  Saved {fold_name}: images {images.shape}, labels {labels.shape}, types {types.shape}")


def prepare_conic_dataset(
    max_samples: int | None = None,
) -> None:
    """Download PanNuke folds and save as CONIC-format .npy arrays.

    Idempotent: if all three folds already have cached arrays, does nothing.
    Set *max_samples* to cap samples per fold (for smoke tests).

    After this, `src/data_utils.get_pannuke(params)` can load directly from disk.
    """
    for fold_name in FOLD_NAMES:
        if max_samples is None and _fold_has_data(fold_name):
            print(f"{fold_name} already cached, skipping.")
            continue

        if max_samples is not None and _fold_has_data(fold_name):
            print(f"{fold_name} cached but --max-samples set, re-downloading.")

        images, labels, types = _process_fold(fold_name, max_samples)
        _save_fold(fold_name, images, labels, types)
        del images, labels, types
        gc.collect()

    print(f"\nCONIC dataset ready at {DATA_DIR}/")
