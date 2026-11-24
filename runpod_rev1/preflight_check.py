"""Pre-flight verification BEFORE running Paket C on RunPod RTX 4090.

Prevents mid-run crashes caused by missing paths, folder naming mismatches,
256px vs 768px confusion, missing checkpoints, insufficient disk space, or missing packages.
"""
import importlib
import os
import shutil
import sys

import pandas as pd
import torch
from PIL import Image

SRC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from common import (get_lc25000_root, get_lunghist_root, resolve_checkpoint_path, resolve_split_csv)


def check_image(root, rel_path, min_side=1):
    clean = str(rel_path).replace("\\", "/").lstrip("/")
    p = os.path.join(root, clean)
    if not os.path.isfile(p):
        raise FileNotFoundError(f"Sample image missing: {p}")
    with Image.open(p) as im:
        w, h = im.size
    if min(w, h) < min_side:
        raise RuntimeError(f"Image {p} has unexpected resolution {(w, h)} (< {min_side}px)")
    return w, h


def main():
    print("=" * 72)
    print("  RUNPOD PAKET C PRE-FLIGHT PATH & INTEGRITY CHECK")
    print("=" * 72)
    errors = []

    # 1. GPU check
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        vram_gb = props.total_memory / (1024 ** 3)
        print(f"[OK] GPU: {props.name} ({vram_gb:.1f} GB VRAM)")
    else:
        errors.append("CUDA GPU not available!")

    # 2. Disk check
    free_gb = shutil.disk_usage(".").free / (1024 ** 3)
    print(f"[OK] Free disk in {os.getcwd()}: {free_gb:.1f} GB")
    if free_gb < 60.0:
        print(f"[WARN] Free disk {free_gb:.1f} GB < 60 GB! Paket C needs ~65 GB free for raw weights + split zips.")

    # 3. Packages (including functional transformers/timm compatibility check)
    req_pkgs = ["timm", "transformers", "safetensors", "umap", "cv2", "scipy", "sklearn", "pandas", "matplotlib", "yaml", "fvcore", "thop"]
    for mod in req_pkgs:
        try:
            m = importlib.import_module(mod)
            if mod == "transformers":
                getattr(m, "AutoImageProcessor")
                getattr(m, "AutoModel")
        except Exception as e:
            errors.append(f"Python import failed ({mod}): {e}")
    print(f"[OK] Python packages checked ({len(req_pkgs)} modules)")

    # 4. All 9 split CSVs
    csvs = [
        "lc25000_dev.csv", "lc25000_test.csv", "lc25000_lung_all.csv",
        "exp2_images.csv", "exp4_groups.csv", "exp4_grouped_split.csv",
        "lunghist700.csv", "folds_patient.csv",
    ]
    resolved_csvs = {}
    for c in csvs:
        rp = resolve_split_csv(c)
        if not os.path.isfile(rp):
            errors.append(f"Missing split CSV: {c} (checked {rp})")
        else:
            resolved_csvs[c] = rp
    print(f"[OK] Split CSVs found: {len(resolved_csvs)}/{len(csvs)}")

    # 5. LC25000 raw 768x768 (WP02, WP06f)
    raw_lc = get_lc25000_root("", require_raw=True)
    if not raw_lc or not os.path.isdir(os.path.join(raw_lc, "lung_aca")):
        errors.append(f"LC25000 768x768 full raw root missing: '{raw_lc}'")
    else:
        try:
            sample_p = pd.read_csv(resolved_csvs["exp4_groups.csv"]).iloc[0]["path"]
            w, h = check_image(raw_lc, sample_p, min_side=512)
            print(f"[OK] LC25000 Raw 768px : {raw_lc} (sample={w}x{h})")
        except Exception as exc:
            errors.append(f"LC25000 Raw 768px check failed: {exc}")

    # 6. LC25000 256px cache (WP03, WP04, WP05)
    cache_lc = get_lc25000_root("", require_raw=False)
    if not cache_lc or not os.path.isdir(os.path.join(cache_lc, "lung_aca")):
        errors.append(f"LC25000 256px full cache missing: '{cache_lc}'")
    else:
        try:
            for csv_key in ["lc25000_dev.csv", "lc25000_test.csv", "exp4_grouped_split.csv"]:
                sp = pd.read_csv(resolved_csvs[csv_key]).iloc[0]["path"]
                check_image(cache_lc, sp, min_side=128)
            print(f"[OK] LC25000 Cache 256px: {cache_lc}")
        except Exception as exc:
            errors.append(f"LC25000 Cache 256px check failed: {exc}")

    # 7. LungHist700 (WP06)
    lh_root = get_lunghist_root("")
    if not lh_root or not os.path.isdir(os.path.join(lh_root, "aca_bd")):
        errors.append(f"LungHist700 root missing (aca_bd not found): '{lh_root}'")
    else:
        try:
            sp = pd.read_csv(resolved_csvs["lunghist700.csv"]).iloc[0]["path"]
            w, h = check_image(lh_root, sp, min_side=200)
            print(f"[OK] LungHist700 root  : {lh_root} (sample={w}x{h})")
        except Exception as exc:
            errors.append(f"LungHist700 check failed: {exc}")

    # 8. All 20 old v2 checkpoints (WP05, WP06a/b/c)
    ckpt_ok = 0
    for grp in ["resnet50_main", "vit_s_main", "clean_resnet50", "clean_vit_s16"]:
        for f in range(5):
            cand = resolve_checkpoint_path(f"old_ckpts/runs/{grp}/fold{f}.pt")
            if os.path.isfile(cand):
                ckpt_ok += 1
            else:
                errors.append(f"Missing v2 checkpoint: {grp}/fold{f}.pt")
    print(f"[OK] Old v2 checkpoints: {ckpt_ok}/20 verified")

    print("=" * 72)
    if errors:
        print("[PREFLIGHT FAILED] Masalah ditemukan:")
        for e in errors:
            print("  -", e)
        sys.exit(1)
    else:
        print("[PREFLIGHT PASSED] Seluruh path, dataset, split CSV, dan 20 checkpoint selaras!")


if __name__ == "__main__":
    main()
