"""HoVer-NeXt single-GPU training on PanNuke.

This is a simplified single-GPU version of the upstream HoVer-NeXt training loop.
All DDP/torchrun code has been removed. Training uses a step-based loop with
AMP, encoder warmup freeze, CosineAnnealingLR, and validation mPQ tracking.
"""

from __future__ import annotations

import os
import random
import sys

import numpy as np
import toml
import torch
from torch.cuda.amp import GradScaler

from src.color_conversion import color_augmentations
from src.data_utils import get_data
from src.focal_loss import FocalCE, FocalLoss
from src.multi_head_unet import freeze_enc, get_model, load_checkpoint, unfreeze_enc
from src.spatial_augmenter import SpatialAugmenter
from src.train_utils import InstanceLoss, save_model, supervised_train_step
from src.validation import validation

random.seed(42)
torch.backends.cudnn.benchmark = True
torch.manual_seed(42)

DEFAULT_CONFIG = "sample_configs/train_pannuke.toml"


def newest(path):
    files = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
    paths = [os.path.join(path, basename) for basename in files]
    return max(paths, key=os.path.getctime)


def train_model(config_path: str = DEFAULT_CONFIG, max_samples: int | None = None):
    """Run HoVer-NeXt training on PanNuke.

    Args:
        config_path: Path to TOML config file.
        max_samples: If set, caps samples per fold (smoke test).
    """
    from data import prepare_conic_dataset

    params = toml.load(config_path)
    params["experiment"] = params["experiment"] + "_" + str(params["fold"])
    params["data_path"] = str(__import__("data").DATA_DIR)

    if max_samples is not None:
        params["batch_size"] = min(params.get("batch_size", 8), 4)
        params["validation_batch_size"] = min(params.get("validation_batch_size", 8), 4)
        params["training_steps"] = min(params.get("training_steps", 200000), 20)
        params["warmup_steps"] = 0
        params["validation_step"] = 10
        params["checkpoint_step"] = 10

    log_dir = os.path.join(params["experiment"], "train")
    os.makedirs(log_dir, exist_ok=True)
    with open(os.path.join(params["experiment"], "params.toml"), "w") as f:
        toml.dump(params, f)

    print("Preparing PanNuke CONIC dataset...")
    prepare_conic_dataset(max_samples=max_samples)

    supervised_training(params, log_dir)


def supervised_training(params, log_dir):
    """Single-GPU training loop."""
    torch.set_num_threads(params["num_workers"])
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    rank = 0
    validation_loss = []

    model = get_model(
        enc=params["encoder"],
        out_channels_cls=params["out_channels_cls"],
        out_channels_inst=params["inst_channels"],
        pretrained=params["pretrained"],
    ).to(device)

    if params.get("checkpoint_path"):
        model, step, best_loss = load_checkpoint(model, params["checkpoint_path"], rank=0)
        params["step"] = step
        validation_loss.append(best_loss)
    model.train()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=params["learning_rate"],
        weight_decay=params["weight_decay"],
        eps=1e-4,
    )

    scaler = GradScaler()

    if params.get("checkpoint_path"):
        optimizer.load_state_dict(
            torch.load(params["checkpoint_path"], map_location="cpu")["optimizer_state_dict"]
        )
        print("Loaded optimizer state dict")

    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=params["training_steps"], eta_min=params["min_learning_rate"]
    )

    if params["use_ema_loss"]:
        ce_loss_fn = FocalCE(num_classes=params["out_channels_cls"]).to(device)
    else:
        ce_loss_fn = FocalLoss(alpha=None, gamma=params["fl_gamma"], reduction="mean").to(device)

    inst_loss_fn = InstanceLoss(params)

    color_aug_fn = color_augmentations(True, s=params["color_scale"], rank=rank)
    fast_aug = SpatialAugmenter(params["aug_params_fast"], random_seed=params["seed"])

    train_dataloaders, validation_dataloader, sz, _, class_names = get_data(params)

    if params.get("step") is not None:
        step = params["step"]
    else:
        step = -1
    print("Start step:", step)
    ep_cnt = 0
    na_steps = []

    freeze_enc(model)
    print(f"Training for {params['training_steps']} steps, starting from step {step+1}")

    while step < params["training_steps"]:
        train_loaders = [iter(x) for x in train_dataloaders]

        for _ in range(sz):
            if step == params["warmup_steps"]:
                print("Warmup steps reached, unfreezing encoder weights...")
                unfreeze_enc(model)

            raw, gt = next(train_loaders[random.randint(0, len(train_loaders) - 1)])
            step += 1

            for param in model.parameters():
                param.grad = None

            loss = supervised_train_step(
                model,
                raw,
                gt,
                fast_aug,
                color_aug_fn,
                inst_loss_fn,
                ce_loss_fn,
                device,
                params,
            )

            if not torch.isfinite(loss):
                na_steps.append(1)
            if len(na_steps) > 10:
                raise ValueError("Too many NaN steps, something is wrong with the model training")

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            scaler.step(optimizer)
            scaler.update()
            lr_scheduler.step()

            if step % params["validation_step"] == 0:
                model.eval()
                with torch.no_grad():
                    val_new = validation(
                        model,
                        validation_dataloader,
                        inst_loss_fn,
                        ce_loss_fn,
                        device,
                        step,
                        nclasses=len(class_names),
                        class_names=class_names,
                        use_amp=params["use_amp"],
                        metric=params["optim_metric"],
                    )
                val_new = val_new.cpu().numpy()
                validation_loss.append(val_new)
                print(f"Step {step}: validation mPQ={val_new:.4f} (best={np.nanmax(validation_loss):.4f})")

                if len(validation_loss) == 0 or val_new >= np.nanmax(validation_loss):
                    print("Save best model")
                    save_model(
                        step,
                        model,
                        optimizer,
                        loss,
                        val_new,
                        os.path.join(log_dir, "best_model"),
                    )
                ep_cnt += 1
                model.train()
                sys.stdout.flush()

            if step % params["checkpoint_step"] == 0:
                save_model(
                    step,
                    model,
                    optimizer,
                    loss,
                    np.nanmax(validation_loss) if validation_loss else 0,
                    os.path.join(log_dir, "checkpoint_step_" + str(step)),
                )
                sys.stdout.flush()

            if step >= params["training_steps"]:
                break

    print(f"\nTraining complete. Best validation mPQ: {np.nanmax(validation_loss) if validation_loss else 'N/A'}")
    print(f"Best model saved at {os.path.join(log_dir, 'best_model')}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train HoVer-NeXt on PanNuke")
    parser.add_argument(
        "--config",
        type=str,
        default=DEFAULT_CONFIG,
        help="Path to TOML config file",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        metavar="N",
        help="Limit samples per fold for quick smoke tests",
    )
    args = parser.parse_args()
    train_model(config_path=args.config, max_samples=args.max_samples)
