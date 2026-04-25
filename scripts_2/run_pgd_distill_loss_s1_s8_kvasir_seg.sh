#!/bin/bash

DATASET_KEY="kvasir_seg"
ROOT_PATH="data/Kvasir-SEG"
EXP_NAME="pgd_kvasir_seg"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/pgd_distill_loss_s1_s8_runner.inc"
