#!/bin/bash

DATASET_KEY="cvc_clinicdb"
ROOT_PATH="data/CVC-ClinicDB"
EXP_NAME="pgd_cvc_clinicdb"

USE_KD_OUTPUT="${USE_KD_OUTPUT:-1}"
USE_SPARSITY="${USE_SPARSITY:-0}"
USE_FEATURE_DISTILL="${USE_FEATURE_DISTILL:-0}"
USE_AUX_LOSS="${USE_AUX_LOSS:-0}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/pgd_s1_s8_runner.inc"
