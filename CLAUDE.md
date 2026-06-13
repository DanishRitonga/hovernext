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
# Training — fine-tune Cellpose-SAM on PanNuke
python main.py train
python main.py train --max-samples 5          # quick smoke test

# Inference
python main.py predict path/to/image.png
python main.py predict path/to/image.png -o output.tiff

# Installed entry point (same as python main.py)
cellpose-skripsi train
cellpose-skripsi predict path/to/image.png
```

There is no test suite.

## Architecture

Four files, no packages:

| File | Role |
|------|------|
| `main.py` | argparse CLI router — delegates to `train.train_model()` or `predict.predict()` |
| `data.py` | PanNuke loading, instance-mask-to-label-map conversion, .npy caching |
| `train.py` | Cellpose CPSAM fine-tuning using `cellpose.train.train_seg()` |
| `predict.py` | Single-image inference via `CellposeModel.eval()` |

### Data flow

**PanNuke** (HuggingFace `RationAI/PanNuke`) has three folds. `prepare_dataset` in `data.py`:

1. **First run** — streams folds from HuggingFace (`streaming=True`), converts instance masks to integer label maps, caches as `.npy` files.
2. **Subsequent runs** — loads cached `.npy` files from disk.

Returns `(train_imgs, train_lbls, val_imgs, val_lbls)` — four lists of numpy arrays passed directly to cellpose's `train_seg()`.

### Data directory layout

```
data/pannuke/
  train/              (fold1)
    000000.npy        (image, uint8, shape 256×256×3)
    000000_label.npy  (label map, int32, shape 256×256)
    ...
  val/                (fold2)
  test/               (fold3)
```

### Training

- Starts from pretrained **CPSAM** weights (downloaded automatically by cellpose).
- Uses `cellpose.train.train_seg()` with: `n_epochs=100`, `lr=1e-5`, `weight_decay=0.1`, `batch_size=1`.
- Model saved to `~/.cellpose/models/cellpose_pannuke.pt` (cellpose's default location).

### Key details

- Cellpose is **instance-only** segmentation — no multi-class support. All nuclei are treated equally regardless of cell type.
- `CELL_TYPES = ["Neoplastic", "Inflammatory", "Connective", "Dead", "Epithelial"]` — defined for reference but not used in training.
- `--max-samples` caps samples per fold for quick smoke tests.
- `predict.py` outputs an integer label map where `0 = background`, `1, 2, ... = instance IDs`.
