# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Environment

Python **3.12** (`requires-python = ">=3.12,<3.13"`). The venv is managed with `uv`.

```bash
uv sync                  # install / sync dependencies
source .venv/bin/activate
```

## Commands

```bash
# Training — HoVer-NeXt on PanNuke (step-based, single GPU)
python main.py train
python main.py train --max-samples 5          # quick smoke test
python main.py train --config sample_configs/train_pannuke.toml

# Inference
python main.py predict path/to/image.png
python main.py predict path/to/image.png -o output.tiff
python main.py predict path/to/image.png --checkpoint path/to/best_model

# Evaluation (mPQ / bPQ on PanNuke test fold)
python main.py evaluate
python main.py evaluate --checkpoint path/to/best_model --max-samples 10

# Installed entry point (same as python main.py)
hovernext-skripsi train
hovernext-skripsi predict path/to/image.png
```

There is no test suite.

## Architecture

Based on [HoVer-NeXt](https://github.com/digitalpathologybern/hover_next_train) — nucleus instance segmentation + classification on PanNuke. Single-GPU (DDP removed from upstream).

### Top-level files

| File | Role |
|------|------|
| `main.py` | argparse CLI router — delegates to `train.train_model()`, `predict.predict()`, or `evaluate.evaluate()` |
| `data.py` | PanNuke → CONIC format converter. Streams from HuggingFace, saves fold-based `.npy` arrays |
| `train.py` | HoVer-NeXt single-GPU training loop (step-based, AMP, encoder warmup freeze, CosineAnnealingLR) |
| `predict.py` | Single-image inference + watershed post-processing |
| `evaluate.py` | Full evaluation on PanNuke test fold using mPQ/bPQ metrics |
| `sample_configs/train_pannuke.toml` | Default training configuration |

### Upstream modules (`src/`)

| Module | Role |
|--------|------|
| `multi_head_unet.py` | Model: `TimmEncoderFixed` (ConvNeXt-V2) → SMP Unet decoder → instance head (5ch) + classification head (6ch) |
| `train_utils.py` | `supervised_train_step()`, `InstanceLoss` (CPV smooth_l1 + 3-class CE), `save_model()` |
| `data_utils.py` | `SliceDataset`, `get_pannuke()`, `parallel_cpvs`, `add_3c_gt_fast`, `inst_to_3c` |
| `validation.py` | `validation()` — computes mPQ during training; `make_instance_segmentation()`, `make_ct()` |
| `post_proc_utils.py` | `process_tile()`, `evaluate()` — threshold optimization, watershed, hole removal |
| `inference_utils.py` | `run_inference()` — TTA inference |
| `metrics.py` | `get_pq()`, `calc_MPQ()` — Lizard-style metrics |
| `spatial_augmenter.py` | `SpatialAugmenter` — mirror, translate, scale, zoom, rotate, shear, elastic |
| `color_conversion.py` | HED color augmentation, percentile normalization |
| `focal_loss.py` | `FocalLoss`, `FocalCE` |
| `constants.py` | `CLASS_NAMES_PANNUKE`, `PANNUKE_FOLDS`, threshold arrays |

### Metrics (`pannuke_metrics_master/`)

| Module | Role |
|--------|------|
| `variant.py` | `get_pannuke_pq()` — PanNuke-specific mPQ/bPQ computation |
| `utils.py` | PanNuke PQ helper functions |

### Data flow

**PanNuke** (HuggingFace `RationAI/PanNuke`) has three folds (fold1, fold2, fold3). `prepare_conic_dataset` in `data.py`:

1. **First run** — streams folds from HuggingFace (`streaming=True`), converts instance masks + categories to CONIC format `(instance_id, class_id)` label maps, saves as `.npy` arrays.
2. **Subsequent runs** — checks for existing `.npy` files and skips if present.

### Data directory layout (CONIC format)

```
data/pannuke_conic/
  images/
    fold1/
      images.npy    (N, 256, 256, 3) uint8
      types.npy     (N,) <U30 tissue-type strings
    fold2/...
    fold3/...
  masks/
    fold1/
      labels.npy    (N, 256, 256, 2) int32 [instance_id, class_id]
    fold2/...
    fold3/...
```

### Training

- **Encoder**: `convnextv2_tiny.fcmae_ft_in22k_in1k` (pretrained from timm).
- **Model**: `MultiHeadModel` — encoder → SMP Unet decoder → two heads:
  - Instance head (5 channels): 2 CPV (center-of-mass vertical/horizontal vectors) + 3-class (background/inside/boundary).
  - Classification head (6 channels): 5 cell types + background.
- **Loss**: `loss_lambda * InstanceLoss + (1 - loss_lambda) * ClassLoss`. InstanceLoss = CPV smooth_l1 + 3-class CE (with label smoothing). ClassLoss = FocalLoss or FocalCE.
- **Training**: step-based (200k steps), AdamW (`lr=1e-4`, `weight_decay=1e-4`), CosineAnnealingLR, AMP fp16.
- **Encoder warmup**: encoder frozen for first 10k steps, then unfrozen.
- **Validation**: every 1000 steps, computes mPQ on validation fold.
- **Checkpoints**: best model saved by mPQ, periodic checkpoints every 10000 steps.
- **Config**: `sample_configs/train_pannuke.toml` — `batch_size=8` (tuned for 12GB VRAM).

### Inference & Post-processing

- Model outputs raw predictions → softmax → threshold-based foreground/seed extraction → watershed → hole removal.
- `process_tile()` handles per-class thresholding using `MIN_THRESHS_PANNUKE` / `MAX_THRESHS_PANNUKE`.
- Output: instance map (`0=bg, 1,2,...=instance IDs`) + class map (`0=bg, 1..5=cell types`).

### Key details

- **5 cell types**: Neoplastic, Inflammatory, Connective, Dead, Epithelial (class indices 1–5; 0 = background).
- **Fold scheme**: `fold=2` means train on fold2, validate on fold1, test on fold3 (see `PANNUKE_FOLDS` in `constants.py`).
- `--max-samples` caps samples per fold for quick smoke tests (adjusts batch_size, steps, etc.).
- Single-GPU only — all DDP/torchrun code removed from upstream.
- `rank` parameters in `src/` modules are CUDA device indices (always `0`), not DDP ranks.
