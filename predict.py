"""HoVer-NeXt single-image inference and post-processing."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image

from data import CELL_TYPES
from src.color_conversion import get_normalize
from src.multi_head_unet import get_model, load_checkpoint
from src.post_proc_utils import process_tile
from src.constants import MIN_THRESHS_PANNUKE, MAX_THRESHS_PANNUKE, CLASS_NAMES_PANNUKE
from src.validation import make_instance_segmentation, make_ct

DEFAULT_ENCODER = "convnextv2_tiny.fcmae_ft_in22k_in1k"
DEFAULT_CHECKPOINT = Path("pannuke_convnextv2_tiny_2/train/best_model")

DEFAULT_FG_THRESH = [0.7, 0.7, 0.7, 0.7, 0.7]
DEFAULT_SEED_THRESH = [0.3, 0.3, 0.3, 0.3, 0.3]
DEFAULT_HOLE_SIZE = 128


def _load_model(checkpoint_path: str | Path, device: torch.device) -> torch.nn.Module:
    model = get_model(
        enc=DEFAULT_ENCODER,
        out_channels_cls=6,
        out_channels_inst=5,
        pretrained=False,
    )
    model, step, best_loss = load_checkpoint(model, str(checkpoint_path), rank=0)
    model.to(device)
    model.eval()
    return model


def predict(
    image_path: str | Path,
    checkpoint_path: str | Path | None = None,
    fg_thresh: list[float] | None = None,
    seed_thresh: list[float] | None = None,
    max_hole_size: int = DEFAULT_HOLE_SIZE,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Run HoVer-NeXt inference on a single image.

    Args:
        image_path: Path to input RGB image (png/jpg/tiff).
        checkpoint_path: Path to trained model checkpoint.
        fg_thresh: Per-class foreground thresholds for post-processing.
        seed_thresh: Per-class seed thresholds for watershed.
        max_hole_size: Max hole size to fill during post-processing.

    Returns:
        instance_map: (H, W) int32 — 0=background, 1,2,...=instance IDs.
        class_map:    (H, W) int32 — 0=background, 1..5=cell types.
        details:      Dict with 'n_instances' and per-class counts.
    """
    if fg_thresh is None:
        fg_thresh = DEFAULT_FG_THRESH
    if seed_thresh is None:
        seed_thresh = DEFAULT_SEED_THRESH
    if checkpoint_path is None:
        checkpoint_path = DEFAULT_CHECKPOINT

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    img = np.array(Image.open(image_path).convert("RGB"), dtype=np.uint8)

    model = _load_model(checkpoint_path, device)

    raw = torch.from_numpy(img.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(device)

    with torch.no_grad():
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device.type == "cuda"):
            out = model(raw)

    out = out.cpu().detach()
    pred_3c = out[0, 2:5].softmax(0).numpy()
    pred_class = out[0, 5:].softmax(0).numpy()

    _, pred_inst, pred_ct = process_tile(
        0,
        pred_3c,
        pred_class,
        np.array(fg_thresh),
        np.array(seed_thresh),
        max_hole_size,
        MIN_THRESHS_PANNUKE,
        MAX_THRESHS_PANNUKE,
        5,
        CLASS_NAMES_PANNUKE,
    )
    pred_inst = pred_inst[..., 0]
    pred_ct = pred_ct[..., 1]

    n_instances = int(pred_inst.max())
    class_counts = {}
    for inst_id in range(1, n_instances + 1):
        cls = int(pred_ct[pred_inst == inst_id][0])
        if 1 <= cls <= len(CELL_TYPES):
            name = CELL_TYPES[cls - 1]
        else:
            name = f"class_{cls}"
        class_counts[name] = class_counts.get(name, 0) + 1

    details = {"n_instances": n_instances, "class_counts": class_counts}
    return pred_inst.astype(np.int32), pred_ct.astype(np.int32), details


def summarize(instance_map: np.ndarray, class_map: np.ndarray, details: dict) -> None:
    n = details.get("n_instances", int(instance_map.max()))
    print(f"Detected {n} nucleus instance(s).")
    cc = details.get("class_counts", {})
    if cc:
        print("Per-class counts:")
        for name, count in sorted(cc.items()):
            print(f"  {name}: {count}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="HoVer-NeXt inference on a single image")
    parser.add_argument("image", help="Path to input image")
    parser.add_argument("-o", "--output", help="Save instance label map as TIFF")
    parser.add_argument("--output-classes", help="Save class map as TIFF")
    parser.add_argument(
        "--checkpoint",
        default=str(DEFAULT_CHECKPOINT),
        help="Path to trained model checkpoint",
    )
    args = parser.parse_args()

    instance_map, class_map, details = predict(args.image, checkpoint_path=args.checkpoint)
    summarize(instance_map, class_map, details)

    if args.output:
        from tifffile import imwrite

        imwrite(args.output, instance_map.astype(np.uint16))
        print(f"Instance masks -> {args.output}")

    if args.output_classes:
        from tifffile import imwrite

        imwrite(args.output_classes, class_map.astype(np.uint8))
        print(f"Class map -> {args.output_classes}")
