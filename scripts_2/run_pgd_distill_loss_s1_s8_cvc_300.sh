#!/bin/bash

DATASET_KEY="cvc_300"
ROOT_PATH="data/CVC-300"
EXP_NAME="pgd_cvc_300"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/pgd_distill_loss_s1_s8_runner.inc"
