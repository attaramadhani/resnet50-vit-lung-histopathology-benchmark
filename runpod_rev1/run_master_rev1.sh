#!/bin/bash
set -e

MODE="${1:-paket_c}"
SUB_ARG="${2:-}"

echo "=========================================================================="
echo "  MASTER PIPELINE RUNPOD REV1 (CMPBUP-D-26-00450)"
echo "  Mode: $MODE $SUB_ARG | Started: $(date -u)"
echo "=========================================================================="

bash runpod_rev1/00_setup_runpod.sh

case "$MODE" in
    paket_c|all_rev1|c|C)
        export PAKET_MODE="C"
        bash runpod_rev1/01_run_wp02.sh
        bash runpod_rev1/02_run_wp03_wp05.sh
        bash runpod_rev1/06_run_wp06_lunghist.sh all
        bash runpod_rev1/03_run_wp04a_multiseed.sh all
        bash runpod_rev1/04_run_wp04b_lr_sweep.sh
        bash runpod_rev1/05_run_wp04c_warmup_cosine.sh
        ;;
    paket_b|core_priority|b|B)
        export PAKET_MODE="B"
        bash runpod_rev1/01_run_wp02.sh
        bash runpod_rev1/02_run_wp03_wp05.sh
        bash runpod_rev1/06_run_wp06_lunghist.sh all
        bash runpod_rev1/03_run_wp04a_multiseed.sh priority
        bash runpod_rev1/05_run_wp04c_warmup_cosine.sh
        ;;
    paket_a|core|a|A)
        export PAKET_MODE="A"
        bash runpod_rev1/01_run_wp02.sh
        bash runpod_rev1/02_run_wp03_wp05.sh
        bash runpod_rev1/06_run_wp06_lunghist.sh all
        bash runpod_rev1/05_run_wp04c_warmup_cosine.sh
        ;;
    wp02)   bash runpod_rev1/01_run_wp02.sh ;;
    wp03|wp05) bash runpod_rev1/02_run_wp03_wp05.sh ;;
    wp04)
        bash runpod_rev1/03_run_wp04a_multiseed.sh "${SUB_ARG:-all}"
        bash runpod_rev1/04_run_wp04b_lr_sweep.sh
        bash runpod_rev1/05_run_wp04c_warmup_cosine.sh
        ;;
    wp04a)  bash runpod_rev1/03_run_wp04a_multiseed.sh "${SUB_ARG:-all}" ;;
    wp04b)  bash runpod_rev1/04_run_wp04b_lr_sweep.sh ;;
    wp04c)  bash runpod_rev1/05_run_wp04c_warmup_cosine.sh ;;
    wp06)   bash runpod_rev1/06_run_wp06_lunghist.sh "${SUB_ARG:-all}" ;;
    wp06a)  bash runpod_rev1/06_run_wp06_lunghist.sh a ;;
    wp06b)  bash runpod_rev1/06_run_wp06_lunghist.sh b ;;
    wp06c)  bash runpod_rev1/06_run_wp06_lunghist.sh c ;;
    wp06d)  bash runpod_rev1/06_run_wp06_lunghist.sh d ;;
    wp06e)  bash runpod_rev1/06_run_wp06_lunghist.sh e ;;
    wp06f)  bash runpod_rev1/06_run_wp06_lunghist.sh f ;;
    preflight) python runpod_rev1/preflight_check.py; exit 0 ;;
    package) export PAKET_MODE="${SUB_ARG:-C}" ;;
    *)
        echo "Unknown mode: $MODE"
        echo "Usage: bash runpod_rev1/run_master_rev1.sh [paket_c|paket_b|paket_a|wp02|wp03|wp04|wp04a [all|priority]|wp04b|wp04c|wp05|wp06 [all|a,b,c,d,e,f]|package [A|B|C]]"
        exit 1
        ;;
esac

python runpod_rev1/package_rev1.py --all --paket "${PAKET_MODE:-C}"
echo -e "\n[DONE] Master Rev1 ($MODE) finished at $(date -u). Ready for sync_s3_runpod.py / sync_from_runpod.ps1!"
