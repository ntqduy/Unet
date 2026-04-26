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
BASE_LR="${BASE_LR:-0.01}"
NUM_WORKERS="${NUM_WORKERS:-0}"
# Keep ResNet152 pretrained by default. Set ENCODER_PRETRAINED=0 only when you
# explicitly want to train the ResNet encoder from scratch.
ENCODER_PRETRAINED="${ENCODER_PRETRAINED:-1}"
# Set FORCE_RETRAIN=1 when an old compatible checkpoint exists but you want a
# fresh run, for example after changing epochs/lr/augmentation.
FORCE_RETRAIN="${FORCE_RETRAIN:-0}"
CONDA_ENV="${CONDA_ENV:-pgdunet}"

if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
  conda activate "$CONDA_ENV"
fi

BASIC_ARGS=(
  --datasets "${DATASETS[@]}"
  --models "${MODELS[@]}"
  --output-root "$OUTPUT_ROOT"
  --device "$DEVICE"
  --epochs "$EPOCHS"
  --batch-size "$BATCH_SIZE"
  --base-lr "$BASE_LR"
  --num-workers "$NUM_WORKERS"
  --encoder-pretrained "$ENCODER_PRETRAINED"
)

if [ "$FORCE_RETRAIN" = "1" ]; then
  BASIC_ARGS+=(--force-retrain)
fi

python run_basic_model.py "${BASIC_ARGS[@]}"
