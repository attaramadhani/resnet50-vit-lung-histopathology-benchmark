#!/bin/bash
set -e
source runpod_rev1/env_exports.sh 2>/dev/null || true
mkdir -p results_rev1/WP02 results_rev1/logs

RAW_LC_ROOT="${LC_RAW_ROOT:-$(cat lc25000_raw_root.txt 2>/dev/null || PYTHONPATH=src python -c 'from common import get_lc25000_root; print(get_lc25000_root("", require_raw=True))')}"
echo "=========================================================================="
echo "  WP02: VALIDASI AUDIT NEAR-DUPLICATE (GAMBAR ASLI 768x768)"
echo "  Root: $RAW_LC_ROOT"
echo "=========================================================================="

if [ -f "results_rev1/WP02/wp02_grid.csv" ] && [ -f "results_rev1/WP02/wp02_pairs.csv" ] && [ "${FORCE_REV1:-0}" != "1" ]; then
    echo "[Skip] WP02 sudah selesai sebelumnya."
else
    python -u scripts/wp02_audit_validation.py \
        --root "$RAW_LC_ROOT" \
        --splits data_splits \
        --out results_rev1/WP02 2>&1 | tee results_rev1/logs/wp02.log
fi

python runpod_rev1/package_rev1.py --wp WP02 --paket "${PAKET_MODE:-C}"
