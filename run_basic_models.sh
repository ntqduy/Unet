#!/bin/bash

set -euo pipefail

# Edit these lists for the exact experiments you want to run.
# Dataset keys must match dataloaders.dataset.list_available_datasets().
IFS=' ' read -r -a DATASETS <<< "${DATASETS:-cvc_300 cvc_clinicdb kvasir_seg etis cvc_colondb}"
IFS=' ' read -r -a MODELS <<< "${MODELS:-unet resunet vnet unetr unet_resnet152}"

OUTPUT_ROOT="${OUTPUT_ROOT:-outputs}"
DEVICE="${DEVICE:-0}"
EPOCHS="${EPOCHS:-50}"
BATCH_SIZE="${BATCH_SIZE:-8}"
CONDA_ENV="${CONDA_ENV:-pgdunet}"

if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
  conda activate "$CONDA_ENV"
fi

python run_basic_model.py \
  --datasets "${DATASETS[@]}" \
  --models "${MODELS[@]}" \
  --output-root "$OUTPUT_ROOT" \
  --device "$DEVICE" \
  --epochs "$EPOCHS" \
  --batch-size "$BATCH_SIZE"
