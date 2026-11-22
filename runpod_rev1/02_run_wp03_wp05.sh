#!/bin/bash
set -e
source runpod_rev1/env_exports.sh 2>/dev/null || true
mkdir -p results_rev1/WP03 results_rev1/WP05 results_rev1/logs

CACHE_ROOT="${LC_CACHE_ROOT:-$(cat cache_root.txt 2>/dev/null || PYTHONPATH=src python -c 'from common import get_lc25000_root; print(get_lc25000_root("", require_raw=False))')}"
G1_PROBE=$(PYTHONPATH=src python -c "from common import resolve_checkpoint_path; print(resolve_checkpoint_path('old_ckpts/runs/clean_resnet50/fold0.pt'))")
CKPT_OLD=$(dirname "$(dirname "$G1_PROBE")")
[ -z "$CKPT_OLD" ] && CKPT_OLD="old_ckpts/runs"

ARCHES_LIST="resnet50 vit_s16"
GROUPED_ARCHES="resnet50,vit_s16"
if [ "${INCLUDE_DEIT:-0}" = "1" ]; then
    ARCHES_LIST="resnet50 vit_s16 deit_s16"
    GROUPED_ARCHES="resnet50,vit_s16,deit_s16"
fi

echo "=========================================================================="
echo "  WP03: MATCHED-RECIPE COMPARISON (60 RUNS) + EXP3 VIT UNFROZEN"
echo "  Resep Identik: AdamW, LR 1e-4, WD 0.05, Warmup 5 + Cosine, Drop 0, Freeze 0"
echo "  Seeds: 42, 1, 2 | Arches: $ARCHES_LIST"
echo "=========================================================================="

for SEED in 42 1 2; do
    for ARCH in $ARCHES_LIST; do
        echo ">>> WP03 Image-Level | arch=$ARCH | seed=$SEED"
        python -u src/train_scenarios.py \
            --data_root "$CACHE_ROOT" \
            --dev_csv data_splits/lc25000_dev.csv \
            --test_csv data_splits/lc25000_test.csv \
            --arch "$ARCH" \
            --opt AdamW \
            --lr 1e-4 \
            --wd 0.05 \
            --epochs 50 \
            --patience 5 \
            --warmup_epochs 5 \
            --sched cosine \
            --freeze_vit 0 \
            --head_dropout 0.0 \
            --seed "$SEED" \
            --out_root results_rev1/WP03/image_level \
            --run_name "wp03_img_${ARCH}_seed${SEED}" 2>&1 | tee -a results_rev1/logs/wp03.log
    done

    echo ">>> WP03 Group-Aware | arches=$GROUPED_ARCHES | seed=$SEED"
    python -u src/train_clean_grouped.py \
        --data_root "$CACHE_ROOT" \
        --csv data_splits/exp4_grouped_split.csv \
        --arches "$GROUPED_ARCHES" \
        --opt AdamW \
        --lr 1e-4 \
        --wd 0.05 \
        --epochs 50 \
        --patience 5 \
        --warmup_epochs 5 \
        --sched cosine \
        --freeze_vit 0 \
        --head_dropout 0.0 \
        --seed "$SEED" \
        --out_root results_rev1/WP03/group_aware \
        --run_prefix wp03_grp 2>&1 | tee -a results_rev1/logs/wp03.log
done

echo -e "\n>>> Exp 3 Compute Benchmark: ViT-S/16 tanpa freezing..."
if [ ! -f "results_rev1/WP03/exp3_vit_s16_unfrozen.json" ] || [ "${FORCE_REV1:-0}" = "1" ]; then
    python -u src/exp3_compute_benchmark.py \
        --arch vit_s16 \
        --freeze_vit 0 \
        --out results_rev1/WP03/exp3_vit_s16_unfrozen.json 2>&1 | tee -a results_rev1/logs/wp03.log
else
    echo "[Skip] exp3_vit_s16_unfrozen.json sudah ada."
fi

echo -e "\n>>> WP05: Ekspor prediksi per gambar G1 & G2 lama..."
if [ ! -f "results_rev1/WP05/wp05_g1_g2_summary.json" ] || [ "${FORCE_REV1:-0}" = "1" ]; then
    python -u scripts/wp05_export_old_g1_g2_preds.py \
        --csv data_splits/exp4_grouped_split.csv \
        --root "$CACHE_ROOT" \
        --ckpt-root "$CKPT_OLD" \
        --out results_rev1/WP05 2>&1 | tee results_rev1/logs/wp05.log
else
    echo "[Skip] WP05 old G1/G2 sudah selesai."
fi

python runpod_rev1/package_rev1.py --wp WP03 --paket "${PAKET_MODE:-C}"
python runpod_rev1/package_rev1.py --wp WP05 --paket "${PAKET_MODE:-C}"
