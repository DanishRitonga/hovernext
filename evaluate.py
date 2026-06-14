"""Evaluate trained HoVer-NeXt model on PanNuke test fold using mPQ and bPQ."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import toml
from tqdm import tqdm

from data import DATA_DIR, CELL_TYPES
from src.constants import (
    CLASS_NAMES_PANNUKE,
    MIN_THRESHS_PANNUKE,
    MAX_THRESHS_PANNUKE,
)
from src.data_utils import SliceDataset, add_3c_gt_fast
from src.inference_utils import run_inference
from src.multi_head_unet import get_model, load_checkpoint
from src.post_proc_utils import evaluate as post_proc_evaluate, get_pp_params
from src.spatial_augmenter import SpatialAugmenter
from src.color_conversion import color_augmentations
from torch.utils.data import DataLoader

from pannuke_metrics_master.variant import get_pannuke_pq

DEFAULT_CONFIG = "sample_configs/train_pannuke.toml"
DEFAULT_CHECKPOINT = Path("pannuke_convnextv2_tiny_2/train/best_model")


def _load_test_data(params):
    """Load test fold images, labels, and tissue types from CONIC arrays."""
    from src.constants import PANNUKE_FOLDS

    fold = params["fold"] - 1
    val_f, test_f = PANNUKE_FOLDS[fold]

    test_fold = test_f + 1

    images = np.load(
        Path(DATA_DIR) / "images" / f"fold{test_fold}" / "images.npy", mmap_mode="r"
    )
    types = np.load(
        Path(DATA_DIR) / "images" / f"fold{test_fold}" / "types.npy", mmap_mode="r"
    )
    labels = np.load(
        Path(DATA_DIR) / "masks" / f"fold{test_fold}" / "labels.npy", mmap_mode="r"
    )

    return np.array(images), np.array(labels), np.array(types)


def evaluate(
    checkpoint_path: str | Path | None = None,
    config_path: str = DEFAULT_CONFIG,
    max_samples: int | None = None,
    fg_thresh: list[float] | None = None,
    seed_thresh: list[float] | None = None,
) -> dict:
    """Run evaluation on PanNuke test fold.

    Args:
        checkpoint_path: Path to trained model checkpoint.
        config_path: Path to TOML config.
        max_samples: Limit test images (for quick tests).
        fg_thresh: Per-class foreground thresholds. If None, tries to load from
                   checkpoint dir, falls back to defaults.
        seed_thresh: Per-class seed thresholds.

    Returns:
        Dict with 'mPQ', 'bPQ', per-class PQ, and tissue-level metrics.
    """
    params = toml.load(config_path)
    if checkpoint_path is None:
        checkpoint_path = DEFAULT_CHECKPOINT

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print("Loading PanNuke test fold ...")
    images, gt_labels, tissue_types = _load_test_data(params)
    if max_samples is not None:
        images = images[:max_samples]
        gt_labels = gt_labels[:max_samples]
        tissue_types = tissue_types[:max_samples]
    print(f"  {len(images)} test images loaded.")

    gt_3c = add_3c_gt_fast(gt_labels.copy())

    test_dataset = SliceDataset(raw=images, labels=gt_3c)
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=params.get("num_workers", 4),
        pin_memory=True,
    )

    print("Loading trained HoVer-NeXt model ...")
    model = get_model(
        enc=params["encoder"],
        out_channels_cls=params["out_channels_cls"],
        out_channels_inst=params["inst_channels"],
        pretrained=False,
    )
    model, step, _ = load_checkpoint(model, str(checkpoint_path), rank=0)
    model.to(device)
    model.eval()

    color_aug_fn = color_augmentations(False, s=params["color_scale"], rank=0)
    fast_aug = SpatialAugmenter(params["aug_params_fast"], random_seed=params["seed"])

    print("Running inference ...", flush=True)
    pred_emb_list, pred_class_list, gt_list, _ = run_inference(
        test_dataloader,
        [model],
        fast_aug,
        color_aug_fn,
        tta=params.get("tta", 16),
        rank=0,
    )
    print(f"Inference done. Shapes: emb={pred_emb_list.shape}, cls={pred_class_list.shape}", flush=True)

    if fg_thresh is None or seed_thresh is None:
        try:
            fg_thresh, seed_thresh = get_pp_params(
                [params["experiment"]],
                str(Path(checkpoint_path).parent),
                eval_metric=params["eval_optim_metric"],
            )
        except Exception:
            print("Could not load threshold params, using defaults")
            fg_thresh = fg_thresh or [0.7] * 5
            seed_thresh = seed_thresh or [0.3] * 5

    print("Post-processing predictions ...")
    nclasses = 5
    results = post_proc_evaluate(
        pred_emb_list,
        pred_class_list,
        None,
        gt_list,
        np.array(fg_thresh),
        np.array(seed_thresh),
        params,
        criterium="pannuke",
        nclasses=nclasses,
        class_names=CLASS_NAMES_PANNUKE,
        types=tissue_types,
    )

    pan_mpq = results.get("optim", [0])[0] if hasattr(results.get("optim", [0]), '__len__') else results.get("optim", 0)
    pan_bpq = results.get("bpq", 0)

    print(f"\n{'='*50}")
    print(f"  HoVer-NeXt Evaluation Results (test fold, {len(images)} images)")
    print(f"{'='*50}")
    print(f"  mPQ = {pan_mpq:.4f}")
    print(f"  bPQ = {pan_bpq:.4f}")
    print(f"{'='*50}")

    return {
        "mPQ": pan_mpq,
        "bPQ": pan_bpq,
        "per_class_pq": results.get("optim", []),
        "tissue_metrics": results.get("tiss", []),
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate HoVer-NeXt on PanNuke test fold")
    parser.add_argument(
        "--checkpoint",
        default=str(DEFAULT_CHECKPOINT),
        help="Path to trained model checkpoint",
    )
    parser.add_argument(
        "--config",
        default=DEFAULT_CONFIG,
        help="Path to TOML config file",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        metavar="N",
        help="Limit test images for quick evaluation",
    )
    args = parser.parse_args()
    evaluate(
        checkpoint_path=args.checkpoint,
        config_path=args.config,
        max_samples=args.max_samples,
    )
