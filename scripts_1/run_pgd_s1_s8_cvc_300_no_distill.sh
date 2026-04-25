#!/bin/bash

DATASET_KEY="cvc_300"
ROOT_PATH="data/CVC-300"
EXP_NAME="pgd_cvc_300"

# No output distillation and no sparsity by default: recovery uses Lseg only.
USE_KD_OUTPUT="${USE_KD_OUTPUT:-0}"
LAMBDA_DISTILL="${LAMBDA_DISTILL:-0}"
USE_SPARSITY="${USE_SPARSITY:-0}"
USE_FEATURE_DISTILL="${USE_FEATURE_DISTILL:-0}"
USE_AUX_LOSS="${USE_AUX_LOSS:-0}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/pgd_s1_s8_runner.inc"
