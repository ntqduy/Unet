# 2D Medical Image Segmentation Workspace

This repository contains a complete 2D segmentation pipeline for CVC-ClinicDB and Kvasir-SEG.
It supports data split management, model training, evaluation, and visualization.

## 1. Main Features

- Unified dataloader that auto-matches image/mask pairs from common folder layouts.
- Multiple segmentation backbones through one model factory:
  - `unet`
  - `unet_resnet152`
  - `resunet`
  - `vnet`
  - `unetr`
- Training with CrossEntropy + Dice loss.
- Validation/testing with Dice and HD95.
- Built-in qualitative outputs (image / ground truth / prediction / panel).
- Dataset utilities for deterministic split generation and statistics reports.

## 2. Project Structure

```text
Code/
|-- train2d.py
|-- test2d.py
|-- requirements.txt
|-- dataloaders/
|   `-- dataset.py
|-- networks/
|   |-- net_factory.py
|   |-- common.py
|   |-- unet.py
|   |-- Unet_restnet.py
|   |-- residual_unet.py
|   |-- VNet.py
|   `-- unetr.py
|-- utils/
|   |-- losses.py
|   |-- val_2d.py
|   `-- visualization.py
|-- analysis_data/
|   |-- generate_splits.py
|   |-- analyze_datasets.py
|   `-- reports/
`-- data/
```

## 3. Environment Setup

Recommended: Python 3.10+

```bash
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows PowerShell
pip install -r requirements.txt
```

If you need a specific CUDA build, install `torch` and `torchvision` first from the official PyTorch index, then install the remaining packages.

## 4. Dataset Layout

Expected roots:

- `data/CVC-ClinicDB`
- `data/Kvasir-SEG`

The dataloader can read common image/mask folder names such as:

- images, image, original
- masks, mask, labels, ground truth

For reproducible experiments, use split manifests:

- `splits/train.txt`
- `splits/val.txt`
- `splits/test.txt`

## 5. Generate Stable Splits

```bash
python analysis_data/generate_splits.py --dataset all --seed 1337
```

Useful options:

- `--dataset {all,cvc,kvasir}`
- `--cvc_val_ratio` (default: 0.1)
- `--cvc_test_ratio` (default: 0.2)
- `--kvasir_val_ratio` (default: 0.1)

## 6. Analyze Dataset Statistics

```bash
python analysis_data/analyze_datasets.py --dataset all
```

Outputs are written to `analysis_data/reports/` (JSON + Markdown summaries).

## 7. Training

Example (Kvasir + UNet):

```bash
python train2d.py \
  --dataset kvasir \
  --root_path data/Kvasir-SEG \
  --model unet \
  --train_split train \
  --val_split val \
  --batch_size 8 \
  --max_iterations 30000
```

Example (CVC + UNet-ResNet152):

```bash
python train2d.py \
  --dataset cvc \
  --root_path data/CVC-ClinicDB \
  --model unet_resnet152 \
  --train_split train \
  --val_split val \
  --encoder_pretrained 1
```

Checkpoints and logs are saved under:

- `logs/model/supervised/<exp>/`

## 8. Testing

```bash
python test2d.py \
  --dataset kvasir \
  --root_path data/Kvasir-SEG \
  --model unet \
  --split test
```

Test outputs:

- `logs/model/supervised/<exp>/predictions/<split>/case_metrics.csv`
- `logs/model/supervised/<exp>/predictions/<split>/metrics_summary.json`
- visualization folders: `image/`, `gt/`, `pred/`, `panel/`

## 9. Important CLI Arguments

- `--root_path`: dataset root path.
- `--dataset`: `cvc`, `cvc_clinicdb`, `kvasir`, `kvasir_seg`, `cyst2d`, `generic`.
- `--model`: one of models listed by `networks/net_factory.py`.
- `--patch_size H W`: default `256 256`.
- `--num_classes`: default `2`.
- `--in_channels`: `1` (grayscale) or `3` (rgb).
- `--gpu`: CUDA visible device id string.

## 10. Reproducibility Notes

- Training supports deterministic mode via `--deterministic 1` and `--seed`.
- Keep split manifests fixed when comparing models.
- Store experiment names with `--exp` to separate checkpoints/results.
