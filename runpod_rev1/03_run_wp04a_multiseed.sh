#!/bin/bash
set -eo pipefail
source runpod_rev1/env_exports.sh 2>/dev/null || true
mkdir -p results_rev1/WP04a results_rev1/logs

MODE="${1:-all}" # 'priority' (S1,S3,S5,S6,S8,S10 + G1,G2) or 'all' (S1-S14 + G1,G2)
CACHE_ROOT="${LC_CACHE_ROOT:-$(cat cache_root.txt 2>/dev/null || PYTHONPATH=src python -c 'from common import get_lc25000_root; print(get_lc25000_root("", require_raw=False))')}"

if [ "$MODE" = "priority" ]; then
    SC_LIST="1,3,5,6,8,10"
else
    SC_LIST="all"
fi

echo "=========================================================================="
echo "  WP04(a): MULTI-SEED REPLICATION (Seeds 1 & 2, Mode: $MODE)"
echo "  Scenarios: $SC_LIST + Group-Aware (G1, G2)"
echo "=========================================================================="

for SEED in 1 2; do
    echo ">>> WP04a Skenario LC25000 ($SC_LIST) | seed=$SEED"
    python -u src/train_scenarios.py \
        --data_root "$CACHE_ROOT" \
        --dev_csv data_splits/lc25000_dev.csv \
        --test_csv data_splits/lc25000_test.csv \
        --scenarios "$SC_LIST" \
        --epochs 50 \
        --patience 5 \
        --seed "$SEED" \
        --no_aliases \
        --out_root results_rev1/WP04a/scenarios 2>&1 | tee -a results_rev1/logs/wp04a.log

    echo ">>> WP04a Group-Aware G1 & G2 | seed=$SEED"
    python -u src/train_clean_grouped.py \
        --data_root "$CACHE_ROOT" \
        --csv data_splits/exp4_grouped_split.csv \
        --epochs 50 \
        --patience 5 \
        --seed "$SEED" \
        --out_root results_rev1/WP04a/grouped 2>&1 | tee -a results_rev1/logs/wp04a.log
done

python runpod_rev1/package_rev1.py --wp WP04a --paket "${PAKET_MODE:-C}"
