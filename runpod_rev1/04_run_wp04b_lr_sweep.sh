#!/bin/bash
set -e
source runpod_rev1/env_exports.sh 2>/dev/null || true
mkdir -p results_rev1/WP04b results_rev1/logs

CACHE_ROOT="${LC_CACHE_ROOT:-$(cat cache_root.txt 2>/dev/null || PYTHONPATH=src python -c 'from common import get_lc25000_root; print(get_lc25000_root("", require_raw=False))')}"
LRS=("1e-5" "3e-5" "1e-4" "3e-4" "1e-3" "3e-3" "1e-2")

echo "=========================================================================="
echo "  WP04(b): 7-POINT LEARNING RATE SWEEP (140 RUNS, SEED=42)"
echo "  LRs: 1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2"
echo "  Optimizers: Primary (Adam/AdamW) & SGD (momentum=0.9)"
echo "=========================================================================="

for ARCH in resnet50 vit_s16; do
    PRIMARY_OPT="Adam"
    [ "$ARCH" = "vit_s16" ] && PRIMARY_OPT="AdamW"

    for OPT_PAIR in "${PRIMARY_OPT}:0.0" "SGD:0.9"; do
        OPT_NAME="${OPT_PAIR%%:*}"
        MOM="${OPT_PAIR##*:}"

        for LR in "${LRS[@]}"; do
            RUN_NAME="sweep_${ARCH}_${OPT_NAME,,}_lr_${LR}"
            echo ">>> WP04b | arch=$ARCH | opt=$OPT_NAME (mom=$MOM) | lr=$LR"
            python -u src/train_scenarios.py \
                --data_root "$CACHE_ROOT" \
                --dev_csv data_splits/lc25000_dev.csv \
                --test_csv data_splits/lc25000_test.csv \
                --arch "$ARCH" \
                --opt "$OPT_NAME" \
                --lr "$LR" \
                --wd 1e-4 \
                --mom "$MOM" \
                --epochs 50 \
                --patience 5 \
                --seed 42 \
                --out_root results_rev1/WP04b \
                --run_name "$RUN_NAME" 2>&1 | tee -a results_rev1/logs/wp04b.log
        done
    done
done

python runpod_rev1/package_rev1.py --wp WP04b --paket "${PAKET_MODE:-C}"
