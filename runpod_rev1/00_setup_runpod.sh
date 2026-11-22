#!/bin/bash
set -e

echo "=========================================================================="
echo "  RUNPOD REV1 ENVIRONMENT & DATASET SETUP (CMPBUP-D-26-00450)"
echo "  Target Hardware: NVIDIA RTX 4090 (or >= 24 GB VRAM GPU)"
echo "=========================================================================="

# 1. Verify persistent /workspace storage
if [ -d "/workspace" ] && [[ "$(pwd)" != /workspace* ]]; then
    echo "[ERROR] Anda tidak sedang berada di dalam /workspace! Direktori saat ini: $(pwd)"
    echo "Di RunPod, container '/' bersifat sementara & kecil. Pindahkan repo ke /workspace/resnet50-vit-lung-histopathology-benchmark lalu jalankan ulang."
    exit 1
fi

df -h .
AVAIL_GB=$(df -BG . 2>/dev/null | awk 'NR==2 {print $4}' | tr -d 'G' || echo "0")
if [ -n "$AVAIL_GB" ] && [ "$AVAIL_GB" -lt 65 ] 2>/dev/null; then
    echo "[WARNING] Ruang disk tersisa ${AVAIL_GB} GB (< 65 GB)!"
    echo "Jika menjalankan WP03 + WP04 penuh (~43 GB checkpoint .pt) + dataset (~2 GB), disarankan volume minimal 80-100 GB."
    echo "Anda tetap bisa lanjut jika hanya menjalankan subset atau langsung sync lalu hapus chunk per WP."
fi

# 2. Redirect all download/wheel caches to workspace so '/' never fills up
CACHE_BASE="/workspace/cache"
[ ! -d "/workspace" ] && CACHE_BASE="$(pwd)/data/cache"
mkdir -p "$CACHE_BASE/kagglehub" "$CACHE_BASE/torch" "$CACHE_BASE/hf" "$CACHE_BASE/pip" results_rev1/logs

export KAGGLEHUB_CACHE_DIR="$CACHE_BASE/kagglehub"
export TORCH_HOME="$CACHE_BASE/torch"
export HF_HOME="$CACHE_BASE/hf"
export PIP_CACHE_DIR="$CACHE_BASE/pip"
export PIP_BREAK_SYSTEM_PACKAGES=1

# Persist exports in env_exports.sh for all runner scripts
cat > runpod_rev1/env_exports.sh <<EOF
export KAGGLEHUB_CACHE_DIR="$CACHE_BASE/kagglehub"
export TORCH_HOME="$CACHE_BASE/torch"
export HF_HOME="$CACHE_BASE/hf"
export PIP_CACHE_DIR="$CACHE_BASE/pip"
export PIP_BREAK_SYSTEM_PACKAGES=1
EOF

# 3. Install dependencies
which unzip >/dev/null 2>&1 || (apt-get update -qq && apt-get install -y -qq unzip p7zip-full unar curl wget rsync >/dev/null 2>&1 || true)
(apt-get install -y -qq unrar >/dev/null 2>&1 || true)
pip install -q --break-system-packages -U "transformers>=4.45" huggingface_hub timm
pip install -q --break-system-packages fvcore thop scikit-learn scipy pandas opencv-python-headless kagglehub safetensors umap-learn matplotlib seaborn pyyaml tqdm boto3

# 4. Prepare LC25000 (original 768x768) + 256px training cache
echo -e "\n>>> Menyiapkan LC25000 (768x768 asli untuk WP02/WP06f + cache 256px untuk WP03/WP04)..."
python - <<'PY'
import os, kagglehub

raw_candidates = [
    os.environ.get("LC25000_RAW_ROOT", ""),
    "/workspace/data/LC25000",
    "data/LC25000",
]
raw_root = ""
for c in raw_candidates:
    if c and os.path.isdir(os.path.join(c, "lung_aca")):
        raw_root = c if c.endswith("/") else c + "/"
        break

if not raw_root:
    print("Mengunduh LC25000 via kagglehub...")
    path = kagglehub.dataset_download("andrewmvd/lung-and-colon-cancer-histopathological-images")
    cand = os.path.join(path, "lung_colon_image_set", "lung_image_sets")
    if not os.path.isdir(os.path.join(cand, "lung_aca")):
        raise RuntimeError(f"Struktur LC25000 tidak ditemukan di {path}")
    raw_root = cand + "/"

with open("lc25000_raw_root.txt", "w", encoding="utf-8") as f:
    f.write(raw_root)
print("[OK] LC25000 768x768 raw root:", raw_root)
PY

RAW_LC_ROOT=$(cat lc25000_raw_root.txt)
if [ ! -d "data/cache_lc25000_256/lung_aca" ]; then
    echo "Membuat cache 256x256 di data/cache_lc25000_256/..."
    PYTHONPATH=src python src/prepare_cache_256.py --raw-root "$RAW_LC_ROOT" --csv data_splits/lc25000_lung_all.csv --cache-root data/cache_lc25000_256/
fi
echo "data/cache_lc25000_256/" > cache_root.txt
echo "[OK] LC25000 256px fast cache siap: data/cache_lc25000_256/"

# 5. Prepare LungHist700
echo -e "\n>>> Menyiapkan LungHist700..."
if [ -d "/workspace/data/LungHist700/aca_bd" ]; then
    echo "/workspace/data/LungHist700/" > lunghist_root.txt
elif [ -d "/workspace/data/LungHist700/data/images/aca_bd" ]; then
    echo "/workspace/data/LungHist700/data/images/" > lunghist_root.txt
else
    PYTHONPATH=src python src/download_lunghist700.py
fi
LH_ROOT=$(PYTHONPATH=src python -c "from common import get_lunghist_root; print(get_lunghist_root())")
echo "$LH_ROOT" > lunghist_root.txt
echo "[OK] LungHist700 siap di: $LH_ROOT"

# 6. Extract old 20 checkpoints if uploaded (supports repo root OR /workspace root from S3)
OLD_CKPT_ZIP=""
[ -f "old_ckpts_for_runpod.zip" ] && OLD_CKPT_ZIP="old_ckpts_for_runpod.zip"
[ -z "$OLD_CKPT_ZIP" ] && [ -f "/workspace/old_ckpts_for_runpod.zip" ] && OLD_CKPT_ZIP="/workspace/old_ckpts_for_runpod.zip"

if [ -n "$OLD_CKPT_ZIP" ] && [ ! -f "old_ckpts/runs/resnet50_main/fold0.pt" ]; then
    echo -e "\n>>> Mengekstrak $OLD_CKPT_ZIP..."
    mkdir -p old_ckpts
    unzip -q -o "$OLD_CKPT_ZIP" -d old_ckpts
fi

MISSING_CKPT=0
for d in resnet50_main vit_s_main clean_resnet50 clean_vit_s16; do
    for f in 0 1 2 3 4; do
        if [ ! -f "old_ckpts/runs/${d}/fold${f}.pt" ] && [ ! -f "runs/${d}/fold${f}.pt" ]; then
            echo "[WARNING] Checkpoint lama belum ada: old_ckpts/runs/${d}/fold${f}.pt (dibutuhkan untuk WP05 & WP06a/b/c)"
            MISSING_CKPT=1
        fi
    done
done
[ "$MISSING_CKPT" -eq 0 ] && echo "[OK] Seluruh 20 checkpoint v2 (G1/G2/S1/S6) terverifikasi tersedia!"

# Create stable canonical symlinks under data/ so every path option resolves identically
mkdir -p data
ln -sfn "$RAW_LC_ROOT" data/LC25000_raw
ln -sfn "$LH_ROOT" data/LungHist700_images

# 7. Provenance, SHA-256 & Pre-Flight Verification
{
  echo "=== RUNPOD REV1 ENVIRONMENT ==="
  date -u
  echo -n "Python: " && python --version
  python -c "import torch; print('PyTorch:', torch.__version__, 'CUDA:', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
  echo "Git commit: $(git rev-parse HEAD 2>/dev/null || echo 'unknown')"
  nvidia-smi 2>/dev/null || true
  pip list | grep -E "torch|timm|scikit|scipy|pandas|numpy|transformers|umap|fvcore|thop|cv2|safetensors"
} > results_rev1/environment_rev1.txt

sha256sum src/*.py scripts/*.py runpod_rev1/*.sh runpod_rev1/*.py > results_rev1/scripts_sha256_rev1.txt 2>/dev/null || true

echo -e "\n>>> Menjalankan Pre-Flight Check (penyelarasan path & integritas dataset/checkpoint)..."
python runpod_rev1/preflight_check.py

echo -e "\n[READY] Setup RunPod Rev1 selesai!"
