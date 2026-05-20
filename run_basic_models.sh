#!/bin/bash

set -euo pipefail

# Edit these lists for the exact experiments you want to run.
# Dataset keys must match dataloaders.dataset.list_available_datasets().
IFS=' ' read -r -a DATASETS <<< "${DATASETS:-cvc_300 cvc_clinicdb kvasir_seg etis cvc_colondb}"
IFS=' ' read -r -a MODELS <<< "${MODELS:-unet resunet vnet unetr unet_resnet152 att_unet r2unet unet_plus_plus}"

OUTPUT_ROOT="${OUTPUT_ROOT:-outputs}"
DEVICE="${DEVICE:-0}"
EPOCHS="${EPOCHS:-50}"
BATCH_SIZE="${BATCH_SIZE:-8}"
BASE_LR="${BASE_LR:-0.01}"
NUM_WORKERS="${NUM_WORKERS:-4}"
EARLY_STOP_PATIENCE="${EARLY_STOP_PATIENCE:-20}"
# Keep supported pretrained encoders enabled by default. Set ENCODER_PRETRAINED=0
# when you explicitly want to train the encoder from scratch.
ENCODER_PRETRAINED="${ENCODER_PRETRAINED:-1}"
# Set FORCE_RETRAIN=1 when an old compatible checkpoint exists but you want a
# fresh run, for example after changing epochs/lr/augmentation.
FORCE_RETRAIN="${FORCE_RETRAIN:-0}"
REUSE_MIN_METRIC="${REUSE_MIN_METRIC:-1e-8}"
VNET_HAS_DROPOUT="${VNET_HAS_DROPOUT:-0}"
VNET_HAS_RESIDUAL="${VNET_HAS_RESIDUAL:-1}"

BASIC_ARGS=(
  --datasets "${DATASETS[@]}"
  --models "${MODELS[@]}"
  --output-root "$OUTPUT_ROOT"
  --device "$DEVICE"
  --epochs "$EPOCHS"
  --batch-size "$BATCH_SIZE"
  --base-lr "$BASE_LR"
  --num-workers "$NUM_WORKERS"
  --early_stop_patience "$EARLY_STOP_PATIENCE"
  --encoder-pretrained "$ENCODER_PRETRAINED"
  --reuse-min-metric "$REUSE_MIN_METRIC"
  --vnet-has-dropout "$VNET_HAS_DROPOUT"
  --vnet-has-residual "$VNET_HAS_RESIDUAL"
)

if [ "$FORCE_RETRAIN" = "1" ]; then
  BASIC_ARGS+=(--force-retrain)
fi

python run_basic_model.py "${BASIC_ARGS[@]}"
