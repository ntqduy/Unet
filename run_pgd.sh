#!/bin/bash
# Backward-compatible default entrypoint. Prefer the explicit dataset scripts:
# run_pgd_cvc_clinicdb.sh, run_pgd_kvasir_seg.sh, run_pgd_etis.sh,
# run_pgd_cvc_colondb.sh, run_pgd_cvc_300.sh.
source "$(dirname "$0")/run_pgd_cvc_clinicdb.sh"
