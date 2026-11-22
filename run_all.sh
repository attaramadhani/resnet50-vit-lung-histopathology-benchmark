#!/bin/bash
set -e

# Logging all outputs to console and run_all.log
exec > >(tee -a run_all.log) 2>&1

echo "=========================================================================="
echo "  MASTER PIPELINE EKSPERIMEN REVISI CMPB UPDATE (RUNPOD NVIDIA RTX 4090)"
echo "  Peneliti: Attala Alif Ramadhani Tri Hida & Dr. Wahyudi Setiawan"
echo "  Mulai pada: $(date)"
echo "=========================================================================="

SCENARIOS_MODE="${1:-all}"
echo "Mode Skenario Pelatihan LC25000: $SCENARIOS_MODE"
echo "(Default: 'all' untuk 14 skenario / 70 runs; opsi: 'main' untuk S1 & S6, 'sgdm' untuk S11-S14)"

# -----------------------------------------------------------------------------
# 0. Pemeriksaan Direktori Kerja & Ruang Penyimpanan Disk (Safety Check)
# -----------------------------------------------------------------------------
echo -e "\n>>> [0/10] Memeriksa Direktori Kerja & Kapasitas Penyimpanan Disk..."
echo "Direktori kerja aktif: $(pwd)"

# Deteksi jika user berjalan di /root (Container root sementara) dan bukan di /workspace
if [ -d "/workspace" ] && [[ "$(pwd)" != /workspace* ]]; then
    echo "=========================================================================="
    echo "  [PERINGATAN DIREKTORI PENYIMPANAN RUNPOD]"
    echo "  Anda saat ini berada di: $(pwd)"
    echo "  Di RunPod, container root '/' atau '/root' berukuran terbatas (20-30 GB)"
    echo "  dan TIDAK PERSISTEN (bisa terhapus jika container di-restart)."
    echo "  Direktori penyimpanan persisten/aman ada di: /workspace"
    echo "  Sangat disarankan menjalankan script ini di: /workspace/eksperimen_atta"
    echo "=========================================================================="
fi

# Cek ruang disk tersedia
df -h .
AVAIL_GB=$(df -BG . 2>/dev/null | awk 'NR==2 {print $4}' | tr -d 'G' || echo "50")
if [ -n "$AVAIL_GB" ] && [ "$AVAIL_GB" -lt 8 ] 2>/dev/null; then
    echo "=========================================================================="
    echo "  [PERINGATAN RUANG DISK MENIPIS]"
    echo "  Ruang disk tersisa hanya ${AVAIL_GB} GB (< 8 GB)!"
    echo "  70 run skenario + dataset + model checkpoint membutuhkan minimal ~10-15 GB."
    echo "  Mohon pastikan direktori kerja berada pada volume persisten /workspace."
    echo "=========================================================================="
fi

# Redirect semua direktori cache download (KaggleHub, Torch, HuggingFace) ke direktori lokal
# agar TIDAK memakan kuota container root /root/.cache
mkdir -p data/cache/kagglehub data/cache/torch data/cache/hf
export KAGGLEHUB_CACHE_DIR="${KAGGLEHUB_CACHE_DIR:-$(pwd)/data/cache/kagglehub}"
export TORCH_HOME="${TORCH_HOME:-$(pwd)/data/cache/torch}"
export HF_HOME="${HF_HOME:-$(pwd)/data/cache/hf}"
echo "Cache diarahkan ke: $(pwd)/data/cache/ (bebas dari risiko /root penuh)"

# -----------------------------------------------------------------------------
# 1. Catat Versi Software & Hardware (Provenance)
# -----------------------------------------------------------------------------
echo -e "\n>>> [1/10] Memeriksa Dependensi & Mencatat Environment..."
which 7z >/dev/null 2>&1 || (apt-get update -qq && apt-get install -y -qq p7zip-full curl wget >/dev/null 2>&1 || true)
pip install --upgrade pip >/dev/null 2>&1 || true
pip install timm fvcore thop scikit-learn scipy pandas opencv-python kagglehub >/dev/null 2>&1 || true

{
  echo "=== ENVIRONMENT PROVENANCE ==="
  date
  echo -n "Python: " && python --version
  echo -n "PyTorch: " && python -c "import torch; print(torch.__version__, 'CUDA available:', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"
  echo "Git commit: $(git rev-parse HEAD 2>/dev/null || echo 'Not a git repo')"
  echo -e "\n=== NVIDIA-SMI ==="
  nvidia-smi 2>/dev/null || echo "nvidia-smi not available"
  echo -e "\n=== PIP PACKAGES ==="
  pip list | grep -E "torch|timm|scikit|scipy|torchvision|fvcore|thop|pandas|numpy"
} > environment.txt

cp environment.txt runpod_software_versions.txt
echo "Environment resmi tercatat di environment.txt"

# -----------------------------------------------------------------------------
# 2. Persiapan Dataset LC25000 & LungHist700 (Deteksi Dinamis & Cache 256px)
# -----------------------------------------------------------------------------
echo -e "\n>>> [2/10] Menyiapkan Dataset LC25000 & LungHist700..."

# Deteksi / Unduh LC25000
python -c "
import kagglehub, os
from common import get_lc25000_root

root = get_lc25000_root()
if not root or not os.path.isdir(os.path.join(root, 'lung_aca')):
    print('Mengunduh dataset LC25000 via kagglehub...')
    path = kagglehub.dataset_download('andrewmvd/lung-and-colon-cancer-histopathological-images')
    full_path = os.path.join(path, 'lung_colon_image_set', 'lung_image_sets') + '/'
    with open('lc25000_root.txt', 'w') as f:
        f.write(full_path)
    print('LC25000 siap di:', full_path)
else:
    print('LC25000 terdeteksi di:', root)
    with open('lc25000_root.txt', 'w') as f:
        f.write(root)
"
LC_ROOT=$(cat lc25000_root.txt)
echo "LC25000 Root Path: $LC_ROOT"

# Pre-processing cache 256px jika belum ada
if [ ! -d "data/cache_lc25000_256/lung_aca" ]; then
    echo "Membuat cache LC25000 256px untuk mengeliminasi disk I/O bottleneck..."
    python src/prepare_cache_256.py --raw-root "$LC_ROOT" --cache-root data/cache_lc25000_256/
fi
CACHE_ROOT=$(python -c "from common import get_lc25000_root; print(get_lc25000_root())")
echo "Active Fast LC25000 Root Path: $CACHE_ROOT"

# Persiapan Dataset LungHist700 (Deteksi Lokal / Unduh Otomatis dari Figshare)
echo "Menyiapkan dataset LungHist700 (deteksi lokal / unduh otomatis dari Figshare DOI: 10.6084/m9.figshare.25459174)..."
python src/download_lunghist700.py
LH_ROOT=$(python -c "from common import get_lunghist_root; print(get_lunghist_root())")
echo "LungHist700 Root Path: $LH_ROOT"

# -----------------------------------------------------------------------------
# 3. Pelatihan 14 Skenario LC25000 (70 Runs 5-Fold CV + 3k Test Set Soft Voting)
# -----------------------------------------------------------------------------
echo -e "\n>>> [3/10] Menjalankan train_scenarios.py (14 Skenario / 70 Run)..."
python src/train_scenarios.py --data_root "$CACHE_ROOT" --scenarios "$SCENARIOS_MODE" --epochs 50 --patience 5
python src/sync_results_aliases.py

# -----------------------------------------------------------------------------
# 4. Exp 4: Audit Duplikasi LC25000 & Group-Aware Re-Training (Section 3.2 & 8)
# -----------------------------------------------------------------------------
echo -e "\n>>> [4/10] Menjalankan Exp 4 (Audit Duplikat & Group-Aware Benchmark)..."
if [ ! -f "exp4_grouped_split.csv" ]; then
    python src/exp4_duplicate_audit.py --csv lc25000_lung_all.csv --root "$CACHE_ROOT" --out exp4 --make-split
else
    echo "[Skip] exp4_grouped_split.csv sudah ada."
fi

if [ ! -f "exp4_clean_benchmark.json" ]; then
    python src/train_clean_grouped.py --data_root "$CACHE_ROOT" --csv exp4_grouped_split.csv --epochs 50 --patience 5
else
    echo "[Skip] exp4_clean_benchmark.json sudah ada."
fi
python src/sync_results_aliases.py

# -----------------------------------------------------------------------------
# 5. Exp 3: Profil Komputasi GPU Murni RTX 4090 (Section 3.5 & Table 4)
# -----------------------------------------------------------------------------
echo -e "\n>>> [5/10] Menjalankan Exp 3 (GPU-only Throughput, Latency, & Real Time)..."
if [ -f "exp3_resnet50.json" ] && [ -f "exp3_vit_s16.json" ]; then
    echo "[Skip] Exp 3 sudah selesai sebelumnya."
else
    python src/exp3_compute_benchmark.py --arch resnet50
    python src/exp3_compute_benchmark.py --arch vit_s16
fi
python src/exp3_compute_benchmark.py --summarize-real-time
python src/sync_results_aliases.py

# -----------------------------------------------------------------------------
# 6. Exp 1: Zero-Shot Ensemble pada Seluruh 691 Gambar LungHist700
# -----------------------------------------------------------------------------
echo -e "\n>>> [6/10] Menjalankan Exp 1 (Zero-Shot Ensemble)..."
if [ -d "$LH_ROOT" ] && [ -d "${LH_ROOT}aca_bd" ]; then
    VIT_DIR="runs/vit_s_main"
    [ ! -d "$VIT_DIR" ] && [ -d "runs/vit_s16_main" ] && VIT_DIR="runs/vit_s16_main"

    if [ -f "exp1_vit_crop768_metrics.json" ]; then
        echo "[Skip] Exp 1 (Zero-shot) sudah selesai sebelumnya."
    else
        python src/exp1_zeroshot_ensemble.py --arch resnet50 --ckpts runs/resnet50_main/fold{0,1,2,3,4}.pt --csv lunghist700.csv --root "$LH_ROOT" --preprocess resize --out exp1_resnet50_main
        python src/exp1_zeroshot_ensemble.py --arch vit_s16  --ckpts ${VIT_DIR}/fold{0,1,2,3,4}.pt  --csv lunghist700.csv --root "$LH_ROOT" --preprocess resize --out exp1_vit_main
        python src/exp1_zeroshot_ensemble.py --arch resnet50 --ckpts runs/resnet50_main/fold{0,1,2,3,4}.pt --csv lunghist700.csv --root "$LH_ROOT" --preprocess crop768 --out exp1_resnet50_crop768
        python src/exp1_zeroshot_ensemble.py --arch vit_s16  --ckpts ${VIT_DIR}/fold{0,1,2,3,4}.pt  --csv lunghist700.csv --root "$LH_ROOT" --preprocess crop768 --out exp1_vit_crop768
    fi
else
    echo "[Info] Folder LungHist700 belum terpasang. Melewatkan Exp 1."
fi
python src/sync_results_aliases.py

# -----------------------------------------------------------------------------
# 7. Exp 2: Faithfulness XAI Benchmark (Section 3.6 & Table 5)
# -----------------------------------------------------------------------------
echo -e "\n>>> [7/10] Menjalankan Exp 2 (Faithfulness XAI K=10 & K=20)..."
VIT_DIR="runs/vit_s_main"
[ ! -d "$VIT_DIR" ] && [ -d "runs/vit_s16_main" ] && VIT_DIR="runs/vit_s16_main"

if [ -f "exp2_resnet50_summary.txt" ] && [ -f "exp2_vit_summary.txt" ]; then
    echo "[Skip] Exp 2 (Faithfulness) sudah selesai sebelumnya."
else
    python src/exp2_faithfulness.py --arch resnet50 --ckpt runs/resnet50_main/fold0.pt --csv lc25000_test.csv --root "$CACHE_ROOT" --n-per-class 50 --out exp2_resnet50
    python src/exp2_faithfulness.py --arch vit_s16  --ckpt ${VIT_DIR}/fold0.pt  --csv lc25000_test.csv --root "$CACHE_ROOT" --n-per-class 50 --out exp2_vit
fi
python src/sync_results_aliases.py

# -----------------------------------------------------------------------------
# 8. Exp 6 & 7: Fine-Tuning LungHist700 (Patient-Grouped & Image Ablation)
# -----------------------------------------------------------------------------
echo -e "\n>>> [8/10] Menjalankan Exp 6 & Exp 7 (LungHist700 Fine-Tuning)..."
if [ -d "$LH_ROOT" ] && [ -d "${LH_ROOT}aca_bd" ]; then
    # Exp 7: Patient-grouped 5-Fold CV
    [ -f "e7_r50_metrics.json" ] || python src/exp7_lunghist_finetune.py --csv lunghist700.csv --root "$LH_ROOT" --arch resnet50 --folds-out folds_patient.csv --out e7_r50
    [ -f "e7_r50_lcinit_metrics.json" ] || python src/exp7_lunghist_finetune.py --csv lunghist700.csv --root "$LH_ROOT" --arch resnet50 --init runs/resnet50_main/fold0.pt --folds folds_patient.csv --out e7_r50_lcinit
    [ -f "e7_vit_metrics.json" ] || python src/exp7_lunghist_finetune.py --csv lunghist700.csv --root "$LH_ROOT" --arch vit_s16 --folds folds_patient.csv --out e7_vit

    # Per-magnification (20x dan 40x untuk ResNet50 dan ViT-S/16)
    [ -f "e7_r50_20x_metrics.json" ] || python src/exp7_lunghist_finetune.py --csv lunghist700.csv --root "$LH_ROOT" --arch resnet50 --folds folds_patient.csv --magnification 20x --out e7_r50_20x
    [ -f "e7_r50_40x_metrics.json" ] || python src/exp7_lunghist_finetune.py --csv lunghist700.csv --root "$LH_ROOT" --arch resnet50 --folds folds_patient.csv --magnification 40x --out e7_r50_40x
    [ -f "e7_vit_20x_metrics.json" ] || python src/exp7_lunghist_finetune.py --csv lunghist700.csv --root "$LH_ROOT" --arch vit_s16 --folds folds_patient.csv --magnification 20x --out e7_vit_20x
    [ -f "e7_vit_40x_metrics.json" ] || python src/exp7_lunghist_finetune.py --csv lunghist700.csv --root "$LH_ROOT" --arch vit_s16 --folds folds_patient.csv --magnification 40x --out e7_vit_40x

    # Exp 6: Image-level Leakage Ablation
    [ -f "e6_r50_image_metrics.json" ] || python src/exp7_lunghist_finetune.py --csv lunghist700.csv --root "$LH_ROOT" --arch resnet50 --split image --out e6_r50_image
    [ -f "e6_vit_image_metrics.json" ] || python src/exp7_lunghist_finetune.py --csv lunghist700.csv --root "$LH_ROOT" --arch vit_s16  --split image --out e6_vit_image
else
    echo "[Info] Folder LungHist700 belum terpasang. Melewatkan Exp 6 & 7."
fi
python src/sync_results_aliases.py

# -----------------------------------------------------------------------------
# 9. Uji Statistik Signifikansi (Section 3.5 & 3.7)
# -----------------------------------------------------------------------------
echo -e "\n>>> [9/10] Menjalankan Uji Statistik Paired t & Clustered McNemar..."
VAL_CSV="per_fold_val.csv"
[ ! -f "$VAL_CSV" ] && [ -f "per_fold.csv" ] && VAL_CSV="per_fold.csv"

if [ -f "$VAL_CSV" ]; then
    python src/exp_stats_paired.py --csv "$VAL_CSV"
fi

if [ -f "e7_r50_oof.csv" ] && [ -f "e7_vit_oof.csv" ]; then
    python src/lunghist_stats.py
fi
python src/sync_results_aliases.py


# -----------------------------------------------------------------------------
# 10. Pengemasan Hasil Akhir & Pembuatan Tabel Manuskrip
# -----------------------------------------------------------------------------
echo -e "\n>>> [10/10] Mengompilasi Tabel Manuskrip & Mengarsipkan Checkpoint..."
python src/package_results.py

echo -e "\nKapasitas disk setelah selesai:"
df -h .

echo "=========================================================================="
echo "  SELURUH PIPELINE EKSPERIMEN REVISI CMPB UPDATE TELAH SELESAI!"
echo "  Waktu selesai: $(date)"
echo "  1. Tabel Manuskrip       : tables_for_manuscript.md"
echo "  2. Arsip Metrik & Log    : hasil_lengkap.zip"
echo "  3. Arsip Checkpoint (.pt): checkpoints.zip"
echo "  4. Log Eksekusi Lengkap  : run_all.log"
echo "=========================================================================="
