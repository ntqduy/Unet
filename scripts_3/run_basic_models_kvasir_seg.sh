#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

DATASET="kvasir_seg"

# Edit MODELS to choose which basic architectures to run.
IFS=' ' read -r -a MODELS <<< "${MODELS:-unet resunet vnet unetr unet_resnet152}"

OUTPUT_ROOT="${OUTPUT_ROOT:-outputs}"
DEVICE="${DEVICE:-0}"
EPOCHS="${EPOCHS:-50}"
BATCH_SIZE="${BATCH_SIZE:-8}"
BASE_LR="${BASE_LR:-0.01}"
NUM_WORKERS="${NUM_WORKERS:-2}"
ENCODER_PRETRAINED="${ENCODER_PRETRAINED:-1}"
FORCE_RETRAIN="${FORCE_RETRAIN:-0}"
CONDA_ENV="${CONDA_ENV:-pgdunet}"

if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
  conda activate "$CONDA_ENV"
fi

BASIC_ARGS=(
  --datasets "$DATASET"
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
