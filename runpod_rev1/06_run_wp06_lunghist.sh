#!/bin/bash
set -e
source runpod_rev1/env_exports.sh 2>/dev/null || true
mkdir -p results_rev1/WP06/wp06a results_rev1/WP06/wp06b results_rev1/WP06/wp06c results_rev1/WP06/wp06d results_rev1/WP06/wp06e results_rev1/WP06/wp06f results_rev1/logs

SUBSET="${1:-all}" # 'all', single letter 'a'..'f', or comma-separated 'a,b,c'
has_sub() {
    [ "$SUBSET" = "all" ] && return 0
    echo ",$SUBSET," | grep -qi ",$1," && return 0
    return 1
}

RAW_LC_ROOT="${LC_RAW_ROOT:-$(cat lc25000_raw_root.txt 2>/dev/null || PYTHONPATH=src python -c 'from common import get_lc25000_root; print(get_lc25000_root("", require_raw=True))')}"
LH_ROOT="${LUNGHIST_ROOT:-$(cat lunghist_root.txt 2>/dev/null || PYTHONPATH=src python -c 'from common import get_lunghist_root; print(get_lunghist_root())')}"
R50_MAIN_CKPT=$(PYTHONPATH=src python -c "from common import resolve_checkpoint_path; print(resolve_checkpoint_path('old_ckpts/runs/resnet50_main/fold0.pt'))")
VIT_MAIN_CKPT=$(PYTHONPATH=src python -c "from common import resolve_checkpoint_path; print(resolve_checkpoint_path('old_ckpts/runs/vit_s_main/fold0.pt'))")
CKPT_OLD=$(dirname "$(dirname "$R50_MAIN_CKPT")")
[ ! -d "$CKPT_OLD" ] && CKPT_OLD="old_ckpts/runs"

echo "=========================================================================="
echo "  WP06: LUNGHIST700 EXTENSION SUITE (Subset: $SUBSET)"
echo "  LC25000 Raw : $RAW_LC_ROOT"
echo "  LungHist700 : $LH_ROOT"
echo "  Old Ckpts   : $CKPT_OLD"
echo "=========================================================================="

# (a) Stain normalization zero-shot
if has_sub a; then
    echo -e "\n>>> [WP06a] Stain Normalization Zero-Shot..."
    if [ -f "results_rev1/WP06/wp06a/wp06a_summary.json" ] && [ "${FORCE_REV1:-0}" != "1" ]; then
        echo "[Skip] WP06a sudah selesai."
    else
        python -u scripts/wp06a_stain_norm_zeroshot.py \
            --lh-csv data_splits/lunghist700.csv \
            --lc-dev-csv data_splits/lc25000_dev.csv \
            --lh-root "$LH_ROOT" \
            --lc-root "$RAW_LC_ROOT" \
            --ckpt-root "$CKPT_OLD" \
            --out results_rev1/WP06/wp06a 2>&1 | tee results_rev1/logs/wp06a.log
    fi
fi

# (b) Stain augmentation (HED jitter sigma=0.05) on folds_patient.csv for 3 models
if has_sub b; then
    echo -e "\n>>> [WP06b] Stain Augmentation (HED jitter sigma=0.05) Fine-Tuning..."
    [ -f "results_rev1/WP06/wp06b/e7_r50_hed_metrics.json" ] || python -u src/exp7_lunghist_finetune.py \
        --csv data_splits/lunghist700.csv --root "$LH_ROOT" --arch resnet50 \
        --folds data_splits/folds_patient.csv --hed_jitter 0.05 \
        --out results_rev1/WP06/wp06b/e7_r50_hed 2>&1 | tee -a results_rev1/logs/wp06b.log

    [ -f "results_rev1/WP06/wp06b/e7_r50_lcinit_hed_metrics.json" ] || python -u src/exp7_lunghist_finetune.py \
        --csv data_splits/lunghist700.csv --root "$LH_ROOT" --arch resnet50 \
        --init "$R50_MAIN_CKPT" \
        --folds data_splits/folds_patient.csv --hed_jitter 0.05 \
        --out results_rev1/WP06/wp06b/e7_r50_lcinit_hed 2>&1 | tee -a results_rev1/logs/wp06b.log

    [ -f "results_rev1/WP06/wp06b/e7_vit_hed_metrics.json" ] || python -u src/exp7_lunghist_finetune.py \
        --csv data_splits/lunghist700.csv --root "$LH_ROOT" --arch vit_s16 \
        --folds data_splits/folds_patient.csv --hed_jitter 0.05 \
        --out results_rev1/WP06/wp06b/e7_vit_hed 2>&1 | tee -a results_rev1/logs/wp06b.log
fi

# (c) ViT-S/16 LC25000-init on LungHist700
if has_sub c; then
    echo -e "\n>>> [WP06c] ViT-S/16 LC25000-init Fine-Tuning..."
    [ -f "results_rev1/WP06/wp06c/e7_vit_lcinit_metrics.json" ] || python -u src/exp7_lunghist_finetune.py \
        --csv data_splits/lunghist700.csv --root "$LH_ROOT" --arch vit_s16 \
        --init "$VIT_MAIN_CKPT" \
        --folds data_splits/folds_patient.csv \
        --out results_rev1/WP06/wp06c/e7_vit_lcinit 2>&1 | tee results_rev1/logs/wp06c.log
fi

# (d) Foundation Model Linear Probe
if has_sub d; then
    echo -e "\n>>> [WP06d] Foundation Model Linear Probe..."
    if [ -f "results_rev1/WP06/wp06d/wp06d_summary.json" ] && [ "${FORCE_REV1:-0}" != "1" ]; then
        echo "[Skip] WP06d sudah selesai."
    else
        UNI_ARG=""
        [ "${INCLUDE_UNI:-0}" = "1" ] && UNI_ARG="--include-uni"
        python -u scripts/wp06d_foundation_linear_probe.py \
            --csv data_splits/lunghist700.csv \
            --folds data_splits/folds_patient.csv \
            --root "$LH_ROOT" \
            --models phikon,ctranspath $UNI_ARG \
            --out results_rev1/WP06/wp06d 2>&1 | tee results_rev1/logs/wp06d.log
    fi
fi

# (e) Stronger Protocol: 512x512 tiling + inner-fold LR sweep + Original 80/10/10 300x400 ResNet50
if has_sub e; then
    echo -e "\n>>> [WP06e] 512x512 Tiling + Inner-fold LR Sweep + Original 80/10/10 Protocol..."
    if [ -f "results_rev1/WP06/wp06e/wp06e_orig_protocol_metrics.json" ] && [ -f "results_rev1/WP06/wp06e/wp06e_tile_vit_s16_metrics.json" ] && [ "${FORCE_REV1:-0}" != "1" ]; then
        echo "[Skip] WP06e sudah selesai."
    else
        python -u scripts/wp06e_stronger_protocol.py \
            --csv data_splits/lunghist700.csv \
            --folds data_splits/folds_patient.csv \
            --root "$LH_ROOT" \
            --mode all \
            --out results_rev1/WP06/wp06e 2>&1 | tee results_rev1/logs/wp06e.log
    fi
fi

# (f) Domain Shift Analysis
if has_sub f; then
    echo -e "\n>>> [WP06f] Domain Shift Analysis (LC25000 vs LungHist700)..."
    if [ -f "results_rev1/WP06/wp06f/wp06f_domain_shift_metrics.json" ] && [ "${FORCE_REV1:-0}" != "1" ]; then
        echo "[Skip] WP06f sudah selesai."
    else
        python -u scripts/wp06f_domain_shift.py \
            --lc-csv data_splits/lc25000_lung_all.csv \
            --lh-csv data_splits/lunghist700.csv \
            --lc-raw-root "$RAW_LC_ROOT" \
            --lh-root "$LH_ROOT" \
            --n-per-dataset 300 \
            --out results_rev1/WP06/wp06f 2>&1 | tee results_rev1/logs/wp06f.log
    fi
fi

python runpod_rev1/package_rev1.py --wp WP06 --paket "${PAKET_MODE:-C}"
