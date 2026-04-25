#!/bin/bash

DATASET_KEY="etis"
ROOT_PATH="data/ETIS"
EXP_NAME="pgd_etis"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/pgd_distill_loss_s1_s8_runner.inc"
