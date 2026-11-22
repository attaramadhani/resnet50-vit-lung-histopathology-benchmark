#!/bin/bash
set -e
source runpod_rev1/env_exports.sh 2>/dev/null || true
mkdir -p results_rev1/WP04c results_rev1/logs

CACHE_ROOT="${LC_CACHE_ROOT:-$(cat cache_root.txt 2>/dev/null || PYTHONPATH=src python -c 'from common import get_lc25000_root; print(get_lc25000_root("", require_raw=False))')}"

echo "=========================================================================="
echo "  WP04(c): WARM-UP (5 EPOCH) + COSINE SCHEDULE (60 RUNS)"
echo "  Scenarios: S1, S3, S6, S8 | Seeds: 42, 1, 2"
echo "=========================================================================="

for SEED in 42 1 2; do
    echo ">>> WP04c S1, S3, S6, S8 | warmup=5 | cosine | seed=$SEED"
    python -u src/train_scenarios.py \
        --data_root "$CACHE_ROOT" \
        --dev_csv data_splits/lc25000_dev.csv \
        --test_csv data_splits/lc25000_test.csv \
        --scenarios 1,3,6,8 \
        --epochs 50 \
        --patience 5 \
        --warmup_epochs 5 \
        --sched cosine \
        --seed "$SEED" \
        --no_aliases \
        --out_root results_rev1/WP04c 2>&1 | tee -a results_rev1/logs/wp04c.log
done

python runpod_rev1/package_rev1.py --wp WP04c --paket "${PAKET_MODE:-C}"
