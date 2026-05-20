#!/bin/bash

DATASET_KEY="cvc_300"
ROOT_PATH="data/CVC-300"
EXP_NAME="pgd_cvc_300"

# Student loss: segmentation only.
LOSS_VARIANTS="seg_only"
USE_KD_OUTPUT="0"
LAMBDA_DISTILL="0"
USE_SPARSITY="0"
USE_FEATURE_DISTILL="0"
USE_AUX_LOSS="0"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/pgd_s1_s12_runner.inc"