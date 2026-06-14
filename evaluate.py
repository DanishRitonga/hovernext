"""Evaluate trained HoVer-NeXt model on PanNuke test fold.

Computes metrics matching LSP-DETR / RayCastED evaluation protocol:
  AJI, AP@0.5, AP@0.7, AP@0.9, AP@0.5:0.05:0.95,
  bPQ, bMPQ, mPQ, mMPQ,
  F1 (centroid, r=12), Precision, Recall.

All instance-matching metrics use Hungarian assignment.
"""

from __future__ import annotations

import resource
from pathlib import Path

import cv2
import numpy as np
import torch
import toml
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm
from torch.utils.data import DataLoader

from data import DATA_DIR
from src.constants import (
    CLASS_NAMES_PANNUKE,
    MIN_THRESHS_PANNUKE,
    MAX_THRESHS_PANNUKE,
)
from src.data_utils import SliceDataset, add_3c_gt_fast
from src.inference_utils import run_inference
from src.multi_head_unet import get_model, load_checkpoint
from src.post_proc_utils import process_tile, get_pp_params
from src.spatial_augmenter import SpatialAugmenter
from src.color_conversion import color_augmentations

PANNUKE_TISSUES = [
    "Adrenal", "BileDuct", "Bladder", "Breast", "Cervix", "Colorectal",
    "Esophagus", "Head&Neck", "Kidney", "Liver", "Lung", "Ovarian",
    "Pancreatic", "Prostate", "Skin", "Stomach", "Testis", "Thyroid", "Uterus",
]

DEFAULT_CONFIG = "sample_configs/train_pannuke.toml"
DEFAULT_CHECKPOINT = Path("pannuke_convnextv2_tiny_2/train/best_model")

NUM_CLASSES = 5
IOU_THRESHOLDS = [round(x, 2) for x in np.arange(0.5, 1.0, 0.05)]


# ---------------------------------------------------------------------------
# Mask helpers
# ---------------------------------------------------------------------------


def inst_map_to_masks(inst_map: np.ndarray) -> list[np.ndarray]:
    """Convert instance map (0=bg, 1..N=instance IDs) to list of binary masks."""
    ids = np.unique(inst_map)
    ids = ids[ids != 0]
    return [(inst_map == i).astype(np.uint8) for i in ids]


def resolve_mask_overlaps(masks: list[np.ndarray]) -> list[np.ndarray]:
    """Resolve overlapping masks using largest-first priority."""
    if len(masks) == 0:
        return masks
    n = len(masks)
    h, w = masks[0].shape
    stack = np.stack(masks)
    areas = stack.sum(axis=(1, 2))
    sorted_indices = np.argsort(-areas)
    occupied = np.zeros((h, w), dtype=bool)
    resolved = [np.zeros((h, w), dtype=np.uint8) for _ in range(n)]
    for idx in sorted_indices:
        resolved[idx] = stack[idx] & ~occupied
        occupied |= resolved[idx].astype(bool)
    return resolved


def mask_iou_matrix(pred_masks: list[np.ndarray], gt_masks: list[np.ndarray]) -> np.ndarray:
    """Compute pairwise mask IoU matrix (N_pred, N_gt)."""
    n_pred = len(pred_masks)
    n_gt = len(gt_masks)
    if n_pred == 0 or n_gt == 0:
        return np.zeros((n_pred, n_gt), dtype=np.float64)
    pred_stack = np.stack(pred_masks).reshape(n_pred, -1).astype(np.float64)
    gt_stack = np.stack(gt_masks).reshape(n_gt, -1).astype(np.float64)
    intersection = pred_stack @ gt_stack.T
    pred_area = pred_stack.sum(axis=1, keepdims=True)
    gt_area = gt_stack.sum(axis=1, keepdims=True)
    union = pred_area + gt_area.T - intersection
    return np.divide(intersection, union, out=np.zeros_like(intersection, dtype=np.float64), where=union > 0)


def compute_aji(pred_masks: list[np.ndarray], gt_masks: list[np.ndarray], iou_threshold: float = 0.5) -> float:
    """Aggregated Jaccard Index with Hungarian matching."""
    if len(gt_masks) == 0 or len(pred_masks) == 0:
        return 0.0
    iou_matrix = mask_iou_matrix(pred_masks, gt_masks)
    row_ind, col_ind = linear_sum_assignment(-iou_matrix)
    valid = iou_matrix[row_ind, col_ind] >= iou_threshold
    match_pred = set(row_ind[valid].tolist())
    match_gt = set(col_ind[valid].tolist())
    total_intersection = 0.0
    total_union = 0.0
    for r, c in zip(row_ind[valid], col_ind[valid]):
        p = pred_masks[r].astype(np.float64)
        g = gt_masks[c].astype(np.float64)
        total_intersection += (p * g).sum()
        total_union += (p + g - p * g).sum()
    for i in range(len(pred_masks)):
        if i not in match_pred:
            total_union += pred_masks[i].astype(np.float64).sum()
    for j in range(len(gt_masks)):
        if j not in match_gt:
            total_union += gt_masks[j].astype(np.float64).sum()
    if total_union == 0:
        return 0.0
    return total_intersection / total_union


def compute_pq_masked(pred_masks, gt_masks, iou_threshold=0.5, mask=None):
    """PQ with optional foreground mask."""
    n_pred = len(pred_masks)
    n_gt = len(gt_masks)
    if n_gt == 0 or n_pred == 0:
        return 0.0, 0.0, 0.0
    if mask is not None:
        pred_masks = [m & mask for m in pred_masks]
        gt_masks = [m & mask for m in gt_masks]
    iou_matrix = mask_iou_matrix(pred_masks, gt_masks)
    row_ind, col_ind = linear_sum_assignment(-iou_matrix)
    valid = iou_matrix[row_ind, col_ind] >= iou_threshold
    tp = valid.sum()
    fp = n_pred - tp
    fn = n_gt - tp
    dq = tp / (tp + 0.5 * fp + 0.5 * fn) if (tp + fp + fn) > 0 else 0.0
    sq = float(iou_matrix[row_ind[valid], col_ind[valid]].mean()) if tp > 0 else 0.0
    pq = sq * dq
    return float(pq), float(sq), float(dq)


def compute_ap(recall, precision):
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0]))
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    indices = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[indices + 1] - mrec[indices]) * mpre[indices + 1]))


# ---------------------------------------------------------------------------
# Tissue name mapping
# ---------------------------------------------------------------------------

_TISSUE_MAP = {
    "Adrenal_gland": 0, "Bile-duct": 1, "Bladder": 2, "Breast": 3, "Cervix": 4,
    "Colon": 5, "Esophagus": 6, "HeadNeck": 7, "Kidney": 8, "Liver": 9,
    "Lung": 10, "Ovarian": 11, "Pancreatic": 12, "Prostate": 13, "Skin": 14,
    "Stomach": 15, "Testis": 16, "Thyroid": 17, "Uterus": 18,
}


def tissue_to_idx(tissue_name: str) -> int:
    return _TISSUE_MAP.get(tissue_name, -1)


# ---------------------------------------------------------------------------
# Streaming metric computation
# ---------------------------------------------------------------------------


def compute_metrics_streaming(
    pred_inst_list: list[np.ndarray],
    pred_cls_list: list[np.ndarray],
    gt_inst_list: list[np.ndarray],
    gt_cls_list: list[np.ndarray],
    tissue_indices: list[int],
    num_classes: int = NUM_CLASSES,
) -> dict:
    """Compute all metrics in a single pass, one image at a time."""
    aji_scores = []
    bpq_scores = []
    bmpq_scores = []
    class_pq = {c: [] for c in range(num_classes)}
    class_mpq = {c: [] for c in range(num_classes)}
    class_bpq = {c: [] for c in range(num_classes)}
    class_ct_tp = [0] * num_classes
    class_ct_fp = [0] * num_classes
    class_ct_fn = [0] * num_classes
    centroid_tp = 0
    centroid_fp = 0
    centroid_fn = 0
    tissue_aji = {t: [] for t in range(len(PANNUKE_TISSUES))}
    tissue_bpq = {t: [] for t in range(len(PANNUKE_TISSUES))}
    tissue_mpq = {t: {c: [] for c in range(num_classes)} for t in range(len(PANNUKE_TISSUES))}

    ap_stats = {c: {t: {"tp": [], "fp": [], "conf": [], "n_gt": 0} for t in IOU_THRESHOLDS} for c in range(num_classes)}

    n_images = len(pred_inst_list)

    for i in range(n_images):
        pred_inst = pred_inst_list[i]
        pred_cls = pred_cls_list[i]
        gt_inst = gt_inst_list[i]
        gt_cls = gt_cls_list[i]

        pred_masks_raw = inst_map_to_masks(pred_inst)
        gt_masks = inst_map_to_masks(gt_inst)
        pred_masks = resolve_mask_overlaps(pred_masks_raw) if pred_masks_raw else []

        pred_ids = [pid for pid in np.unique(pred_inst) if pid != 0]
        gt_ids = [gid for gid in np.unique(gt_inst) if gid != 0]

        pred_centroids = np.array(
            [np.array(np.where(pred_inst == pid)).mean(axis=1)[::-1] for pid in pred_ids]
        ) if pred_ids else np.zeros((0, 2))
        gt_centroids = np.array(
            [np.array(np.where(gt_inst == gid)).mean(axis=1)[::-1] for gid in gt_ids]
        ) if gt_ids else np.zeros((0, 2))

        pred_cls_per_inst = np.array([int(pred_cls[pred_inst == pid][0]) for pid in pred_ids]) if pred_ids else np.array([], dtype=int)
        gt_cls_per_inst = np.array([int(gt_cls[gt_inst == gid][0]) for gid in gt_ids]) if gt_ids else np.array([], dtype=int)

        aji_val = compute_aji(pred_masks, gt_masks)
        aji_scores.append(aji_val)

        if len(pred_masks) > 0:
            pred_binary = np.stack(pred_masks).max(axis=0).astype(np.uint8)
        else:
            pred_binary = np.zeros_like(pred_inst, dtype=np.uint8)
        if len(gt_masks) > 0:
            gt_binary = np.stack(gt_masks).max(axis=0).astype(np.uint8)
        else:
            gt_binary = np.zeros_like(gt_inst, dtype=np.uint8)

        bpq, _, _ = compute_pq_masked([pred_binary], [gt_binary])
        bpq_scores.append(bpq)

        if gt_binary.sum() > 0:
            fg = gt_binary > 0
            bmpq, _, _ = compute_pq_masked([pred_binary], [gt_binary], mask=fg)
        else:
            bmpq = 0.0
        bmpq_scores.append(bmpq)

        class_pq_img = [0.0] * num_classes
        gt_idx_counts = [0] * num_classes
        pred_idx_counts = [0] * num_classes
        for cls_id in range(num_classes):
            pred_idx = [j for j, c in enumerate(pred_cls_per_inst) if c == cls_id + 1]
            gt_idx = [j for j, c in enumerate(gt_cls_per_inst) if c == cls_id + 1]
            gt_idx_counts[cls_id] = len(gt_idx)
            pred_idx_counts[cls_id] = len(pred_idx)

            pred_cls_masks = [pred_masks[j] for j in pred_idx] if pred_idx else []
            gt_cls_masks = [gt_masks[j] for j in gt_idx] if gt_idx else []

            pq, _, _ = compute_pq_masked(pred_cls_masks, gt_cls_masks)
            class_pq[cls_id].append(pq)
            class_pq_img[cls_id] = pq

            if len(gt_masks) > 0:
                gt_any = np.stack(gt_masks).max(axis=0).astype(np.uint8)
                fg = gt_any > 0
                mpq, _, _ = compute_pq_masked(pred_cls_masks, gt_cls_masks, mask=fg)
            else:
                mpq = 0.0
            class_mpq[cls_id].append(mpq)

            if pred_idx and gt_idx:
                cls_pred_binary = np.stack([pred_masks[j] for j in pred_idx]).max(axis=0).astype(np.uint8)
                cls_gt_binary = np.stack([gt_masks[j] for j in gt_idx]).max(axis=0).astype(np.uint8)
                cls_bpq, _, _ = compute_pq_masked([cls_pred_binary], [cls_gt_binary])
                class_bpq[cls_id].append(cls_bpq)

        # AP (per-class Hungarian matching on mask IoU)
        for cls_id in range(num_classes):
            cls_pred_masks = [pred_masks[j] for j in range(len(pred_masks)) if j < len(pred_cls_per_inst) and pred_cls_per_inst[j] == cls_id + 1]
            cls_gt_masks = [gt_masks[j] for j in range(len(gt_masks)) if j < len(gt_cls_per_inst) and gt_cls_per_inst[j] == cls_id + 1]
            n_pred_cls = len(cls_pred_masks)
            n_gt_cls = len(cls_gt_masks)

            for t in IOU_THRESHOLDS:
                ap_stats[cls_id][t]["n_gt"] += n_gt_cls

            if n_pred_cls > 0 and n_gt_cls > 0:
                iou_mat = mask_iou_matrix(cls_pred_masks, cls_gt_masks)
                row_ind, col_ind = linear_sum_assignment(-iou_mat)
                matched_iou = iou_mat[row_ind, col_ind]
            else:
                row_ind = np.array([], dtype=int)
                matched_iou = np.array([], dtype=float)

            for t in IOU_THRESHOLDS:
                if n_gt_cls > 0:
                    valid = matched_iou >= t
                    matched_pred = set(row_ind[valid].tolist())
                else:
                    matched_pred = set()

                for pi in range(n_pred_cls):
                    is_tp = pi in matched_pred
                    ap_stats[cls_id][t]["conf"].append(1.0)
                    ap_stats[cls_id][t]["tp"].append(is_tp)
                    ap_stats[cls_id][t]["fp"].append(not is_tp)

        # Centroid F1 (Hungarian, r=12)
        n_pred = len(pred_centroids)
        n_gt = len(gt_centroids)
        if n_pred > 0 and n_gt > 0:
            dist_matrix = np.linalg.norm(pred_centroids[:, None, :] - gt_centroids[None, :, :], axis=2)
            row_ind, col_ind = linear_sum_assignment(dist_matrix)
            tp = int((dist_matrix[row_ind, col_ind] <= 12).sum())
        elif n_pred > 0:
            tp = 0
        else:
            tp = 0
        centroid_tp += tp
        centroid_fp += n_pred - tp
        centroid_fn += n_gt - tp

        # Per-class centroid F1 (Hungarian, r=12)
        if n_pred > 0 and n_gt > 0:
            for cls_id in range(num_classes):
                p_idx = [j for j in range(n_pred) if pred_cls_per_inst[j] == cls_id + 1]
                g_idx = [j for j in range(n_gt) if gt_cls_per_inst[j] == cls_id + 1]
                if not p_idx and not g_idx:
                    continue
                if p_idx and g_idx:
                    p_pts = pred_centroids[p_idx]
                    g_pts = gt_centroids[g_idx]
                    dm = np.linalg.norm(p_pts[:, None, :] - g_pts[None, :, :], axis=2)
                    ri2, ci2 = linear_sum_assignment(dm)
                    ctp = int((dm[ri2, ci2] <= 12).sum())
                else:
                    ctp = 0
                class_ct_tp[cls_id] += ctp
                class_ct_fp[cls_id] += len(p_idx) - ctp
                class_ct_fn[cls_id] += len(g_idx) - ctp

        tissue = tissue_indices[i]
        if 0 <= tissue < len(PANNUKE_TISSUES):
            tissue_aji[tissue].append(aji_val)
            tissue_bpq[tissue].append(bpq)
            for cls_id in range(num_classes):
                if gt_idx_counts[cls_id] > 0 or pred_idx_counts[cls_id] > 0:
                    tissue_mpq[tissue][cls_id].append(class_pq_img[cls_id])

        if (i + 1) % 500 == 0:
            mem_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
            print(f"  Processed {i + 1}/{n_images} images (RSS={mem_mb:.0f}MB)", flush=True)

    mean_aji = np.mean(aji_scores) if aji_scores else 0.0
    mean_bpq = np.mean(bpq_scores) if bpq_scores else 0.0
    mean_bmpq = np.mean(bmpq_scores) if bmpq_scores else 0.0

    mpq_values = []
    mmpq_values = []
    for c in range(num_classes):
        valid_pq = [v for v in class_pq[c] if v > 0]
        valid_mpq = [v for v in class_mpq[c] if v > 0]
        if valid_pq:
            mpq_values.append(np.mean(valid_pq))
        if valid_mpq:
            mmpq_values.append(np.mean(valid_mpq))
    mean_mpq = np.mean(mpq_values) if mpq_values else 0.0
    mean_mmpq = np.mean(mmpq_values) if mmpq_values else 0.0

    ap_results = {}
    for t in IOU_THRESHOLDS:
        aps = []
        for cls_id in range(num_classes):
            stats = ap_stats[cls_id][t]
            n_gt = stats["n_gt"]
            if n_gt == 0:
                continue
            confs = np.array(stats["conf"])
            tps = np.array(stats["tp"])
            fps = np.array(stats["fp"])
            if len(confs) == 0:
                aps.append(0.0)
                continue
            order = np.argsort(-confs)
            tps = tps[order]
            fps = fps[order]
            cum_tp = np.cumsum(tps)
            cum_fp = np.cumsum(fps)
            precision = cum_tp / (cum_tp + cum_fp)
            recall = cum_tp / n_gt
            aps.append(compute_ap(recall, precision))
        ap_results[t] = {"AP": np.mean(aps) if aps else 0.0}

    prec = centroid_tp / (centroid_tp + centroid_fp) if (centroid_tp + centroid_fp) > 0 else 0.0
    rec = centroid_tp / (centroid_tp + centroid_fn) if (centroid_tp + centroid_fn) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

    # Per-class PQ means
    per_class_pq = {}
    per_class_bpq = {}
    per_class_centroid = {}
    for c in range(num_classes):
        valid_pq = [v for v in class_pq[c] if v > 0]
        valid_bpq = [v for v in class_bpq[c] if v > 0]
        per_class_pq[CLASS_NAMES_PANNUKE[c]] = float(np.mean(valid_pq)) if valid_pq else 0.0
        per_class_bpq[CLASS_NAMES_PANNUKE[c]] = float(np.mean(valid_bpq)) if valid_bpq else 0.0
        ctp = class_ct_tp[c]
        cfp = class_ct_fp[c]
        cfn = class_ct_fn[c]
        cp = ctp / (ctp + cfp) if (ctp + cfp) > 0 else 0.0
        cr = ctp / (ctp + cfn) if (ctp + cfn) > 0 else 0.0
        cf = 2 * cp * cr / (cp + cr) if (cp + cr) > 0 else 0.0
        per_class_centroid[CLASS_NAMES_PANNUKE[c]] = {"precision": cp, "recall": cr, "f1": cf}

    return {
        "aji": mean_aji,
        "bpq": mean_bpq,
        "bmpq": mean_bmpq,
        "mpq": mean_mpq,
        "mmpq": mean_mmpq,
        "ap": ap_results,
        "centroid": {"precision": prec, "recall": rec, "f1": f1},
        "per_class_pq": per_class_pq,
        "per_class_bpq": per_class_bpq,
        "per_class_centroid": per_class_centroid,
        "tissue_aji": tissue_aji,
        "tissue_bpq": tissue_bpq,
        "tissue_mpq": tissue_mpq,
    }


# ---------------------------------------------------------------------------
# Printing helpers
# ---------------------------------------------------------------------------


def _print_tissue_breakdown(metrics: dict, tissue_indices: list[int]) -> None:
    n_tissues = len(PANNUKE_TISSUES)
    t_aji = metrics["tissue_aji"]
    t_bpq = metrics["tissue_bpq"]
    t_mpq = metrics["tissue_mpq"]

    w = 85
    sep = "=" * w
    print(f"\n{sep}")
    print("  Tissue Type Breakdown")
    print(sep)
    print(f"  {'Group':<14} {'Imgs':>5} {'AJI':>7} {'bPQ':>7} {'mPQ':>7}")
    print(f"  {'-'*14} {'-'*5} {'-'*7} {'-'*7} {'-'*7}")

    img_counts = [0] * n_tissues
    for ti in tissue_indices:
        if 0 <= ti < n_tissues:
            img_counts[ti] += 1

    for g in range(n_tissues):
        aji_arr = np.array(t_aji.get(g, []))
        bpq_arr = np.array(t_bpq.get(g, []))
        if len(aji_arr) == 0 and len(bpq_arr) == 0:
            continue
        aji_m = float(np.mean(aji_arr)) if len(aji_arr) else 0.0
        bpq_m = float(np.mean(bpq_arr)) if len(bpq_arr) else 0.0
        mpq_vals = []
        for c in range(NUM_CLASSES):
            v = t_mpq.get(g, {}).get(c, [])
            vp = [x for x in v if x > 0]
            if vp:
                mpq_vals.append(float(np.mean(vp)))
        mpq_m = float(np.mean(mpq_vals)) if mpq_vals else 0.0
        print(f"  {PANNUKE_TISSUES[g]:<14} {img_counts[g]:>5} {aji_m:>7.4f} {bpq_m:>7.4f} {mpq_m:>7.4f}")

    print(sep)


def _print_nuclei_breakdown(metrics: dict) -> None:
    per_class_pq = metrics["per_class_pq"]
    per_class_bpq = metrics["per_class_bpq"]
    per_class_ct = metrics["per_class_centroid"]
    w = 70
    print(f"\n{'='*w}")
    print("  Nuclei Class Breakdown")
    print(f"{'='*w}")
    print(f"  {'Class':<16} {'bPQ':>7} {'mPQ':>7} {'Prec':>7} {'Recall':>7} {'F1':>7}")
    print(f"  {'-'*16} {'-'*7} {'-'*7} {'-'*7} {'-'*7} {'-'*7}")
    for name in CLASS_NAMES_PANNUKE:
        pq = per_class_pq.get(name, 0.0)
        bpq = per_class_bpq.get(name, 0.0)
        ct = per_class_ct.get(name, {})
        p = ct.get("precision", 0.0)
        r = ct.get("recall", 0.0)
        f1 = ct.get("f1", 0.0)
        print(f"  {name.capitalize():<16} {bpq:>7.4f} {pq:>7.4f} {p:>7.4f} {r:>7.4f} {f1:>7.4f}")
    print(f"{'='*w}")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def _load_test_data(params):
    from src.constants import PANNUKE_FOLDS

    fold = params["fold"] - 1
    val_f, test_f = PANNUKE_FOLDS[fold]
    test_fold = test_f + 1

    images = np.load(Path(DATA_DIR) / "images" / f"fold{test_fold}" / "images.npy", mmap_mode="r")
    types = np.load(Path(DATA_DIR) / "images" / f"fold{test_fold}" / "types.npy", mmap_mode="r")
    labels = np.load(Path(DATA_DIR) / "masks" / f"fold{test_fold}" / "labels.npy", mmap_mode="r")

    return np.array(images), np.array(labels), np.array(types)


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------


def evaluate(
    checkpoint_path: str | Path | None = None,
    config_path: str = DEFAULT_CONFIG,
    max_samples: int | None = None,
    fg_thresh: list[float] | None = None,
    seed_thresh: list[float] | None = None,
) -> dict:
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

    print("Post-processing predictions ...", flush=True)
    nclasses = NUM_CLASSES
    if params["dataset"] == "pannuke":
        min_threshs = [0, 0, 0, 0, 0]
    else:
        min_threshs = [0] * nclasses
    max_threshs = MAX_THRESHS_PANNUKE

    pred_inst_list = []
    pred_cls_list = []
    for i in tqdm(range(len(pred_emb_list)), desc="post-processing"):
        pred_3c = pred_emb_list[i]
        pred_class = pred_class_list[i]
        ri, pred_inst, pred_reg = process_tile(
            i,
            pred_3c,
            pred_class,
            np.array(fg_thresh),
            np.array(seed_thresh),
            params["max_hole_size"],
            min_threshs,
            max_threshs,
            nclasses,
            CLASS_NAMES_PANNUKE,
        )
        pred_inst_list.append(pred_inst[..., 0])
        pred_cls_list.append(pred_inst[..., 1])

    print("Computing metrics (streaming) ...", flush=True)
    gt_inst_list = [gt_list[i][..., 0] for i in range(len(gt_list))]
    gt_cls_list = [gt_list[i][..., 1] for i in range(len(gt_list))]
    tissue_indices = [int(t) for t in tissue_types]

    metrics = compute_metrics_streaming(
        pred_inst_list,
        pred_cls_list,
        gt_inst_list,
        gt_cls_list,
        tissue_indices,
        num_classes=nclasses,
    )

    ap_results = metrics["ap"]
    ap50 = ap_results.get(0.5, {}).get("AP", 0.0)
    ap70 = ap_results.get(0.7, {}).get("AP", 0.0)
    ap90 = ap_results.get(0.9, {}).get("AP", 0.0)
    ap50_95 = np.mean([ap_results[t]["AP"] for t in sorted(ap_results.keys())])
    f12 = metrics["centroid"]

    n_params = sum(p.numel() for p in model.parameters())
    params_m = n_params / 1e6

    print(f"\n{'='*60}")
    print("PanNuke Fold3 Evaluation Results (LSP-DETR Protocol)")
    print(f"{'='*60}")
    print(f"{'Metric':<25} {'Value':>12}")
    print(f"{'-'*37}")
    print(f"{'AJI':<25} {metrics['aji']:>12.4f}")
    print(f"{'AP@0.5':<25} {ap50:>12.4f}")
    print(f"{'AP@0.7':<25} {ap70:>12.4f}")
    print(f"{'AP@0.9':<25} {ap90:>12.4f}")
    print(f"{'AP@0.5:0.05:0.95':<25} {ap50_95:>12.4f}")
    print(f"{'bPQ':<25} {metrics['bpq']:>12.4f}")
    print(f"{'bMPQ':<25} {metrics['bmpq']:>12.4f}")
    print(f"{'mPQ':<25} {metrics['mpq']:>12.4f}")
    print(f"{'mMPQ':<25} {metrics['mmpq']:>12.4f}")
    print(f"{'F1 (centroid, r=12)':<25} {f12['f1']:>12.4f}")
    print(f"{'Precision (centroid)':<25} {f12['precision']:>12.4f}")
    print(f"{'Recall (centroid)':<25} {f12['recall']:>12.4f}")
    print(f"{'Params (M)':<25} {params_m:>12.2f}")
    print(f"{'='*37}")

    print(f"\nImages evaluated: {len(pred_inst_list)}")
    print(f"Total predictions: {sum(len(np.unique(p[p != 0])) for p in pred_inst_list)}")
    print(f"Total GT instances: {sum(len(np.unique(g[g != 0])) for g in gt_inst_list)}")

    _print_tissue_breakdown(metrics, tissue_indices)
    _print_nuclei_breakdown(metrics)

    return {
        "aji": metrics["aji"],
        "ap50": ap50,
        "ap70": ap70,
        "ap90": ap90,
        "ap50_95": ap50_95,
        "bpq": metrics["bpq"],
        "bmpq": metrics["bmpq"],
        "mpq": metrics["mpq"],
        "mmpq": metrics["mmpq"],
        "f1": f12["f1"],
        "precision": f12["precision"],
        "recall": f12["recall"],
        "per_class_pq": metrics["per_class_pq"],
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
