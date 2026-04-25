#!/bin/bash

DATASET_KEY="cvc_clinicdb"
ROOT_PATH="data/CVC-ClinicDB"
EXP_NAME="pgd_cvc_clinicdb"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/pgd_distill_loss_s1_s8_runner.inc"
