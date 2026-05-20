#!/bin/bash

DATASET_KEY="etis"
ROOT_PATH="data/ETIS-LaribPolypDB"
EXP_NAME="pgd_etis"

# Student loss: segmentation + teacher output distillation/KD.
LOSS_VARIANTS="seg_kd"
USE_KD_OUTPUT="1"
LAMBDA_DISTILL="${LAMBDA_DISTILL:-0.3}"
USE_SPARSITY="0"
USE_FEATURE_DISTILL="0"
USE_AUX_LOSS="0"
DISTILL_LOSS_METHOD="${DISTILL_LOSS_METHOD:-mse}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/pgd_s1_s12_runner.inc"