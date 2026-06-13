"""CLI entry point for hovernext-skripsi."""

from __future__ import annotations

import argparse

from data import CELL_TYPES
from train import DEFAULT_CONFIG


def cmd_train(args: argparse.Namespace) -> None:
    from train import train_model

    train_model(
        config_path=args.config,
        max_samples=args.max_samples,
    )


def cmd_predict(args: argparse.Namespace) -> None:
    from predict import predict, summarize

    instance_map, class_map, details = predict(
        args.image,
        checkpoint_path=args.checkpoint,
    )
    summarize(instance_map, class_map, details)

    if args.output:
        from tifffile import imwrite

        imwrite(args.output, instance_map.astype(np.uint16))
        print(f"Instance masks -> {args.output}")

    if args.output_classes:
        from tifffile import imwrite

        imwrite(args.output_classes, class_map.astype(np.uint8))
        print(f"Class map -> {args.output_classes}")


def cmd_evaluate(args: argparse.Namespace) -> None:
    from evaluate import evaluate

    evaluate(
        checkpoint_path=args.checkpoint,
        config_path=args.config,
        max_samples=args.max_samples,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="hovernext-skripsi",
        description="HoVer-NeXt nucleus segmentation & classification on PanNuke",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    def _positive_int(v: str) -> int:
        n = int(v)
        if n < 1:
            raise argparse.ArgumentTypeError(f"must be >= 1, got {n}")
        return n

    # ── train ──
    t = sub.add_parser("train", help="Train HoVer-NeXt on PanNuke")
    t.add_argument(
        "--config",
        type=str,
        default=DEFAULT_CONFIG,
        help="Path to TOML config file",
    )
    t.add_argument(
        "--max-samples",
        type=_positive_int,
        default=None,
        metavar="N",
        help="Cap samples per fold for quick smoke tests",
    )

    # ── predict ──
    p = sub.add_parser("predict", help="Run inference on a single image")
    p.add_argument("image", help="Path to input image")
    p.add_argument("--output", "-o", metavar="FILE", help="Save instance map as TIFF")
    p.add_argument("--output-classes", metavar="FILE", help="Save class map as TIFF")
    p.add_argument(
        "--checkpoint",
        default=None,
        help="Path to trained model checkpoint",
    )

    # ── evaluate ──
    e = sub.add_parser("evaluate", help="Evaluate on PanNuke test fold (mPQ / bPQ)")
    e.add_argument(
        "--config",
        type=str,
        default=DEFAULT_CONFIG,
        help="Path to TOML config file",
    )
    e.add_argument(
        "--checkpoint",
        default=None,
        help="Path to trained model checkpoint",
    )
    e.add_argument(
        "--max-samples",
        type=_positive_int,
        default=None,
        metavar="N",
        help="Cap test samples for quick evaluation",
    )

    args = parser.parse_args()

    if args.command == "train":
        cmd_train(args)
    elif args.command == "predict":
        cmd_predict(args)
    elif args.command == "evaluate":
        cmd_evaluate(args)


if __name__ == "__main__":
    main()
