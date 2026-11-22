"""WP06(d) - Pathology Foundation Model Linear Probes on LungHist700.

Evaluates frozen pathology-specific foundation model embeddings on the exact
patient-grouped 5-fold split (data_splits/folds_patient.csv) and exact eval
preprocessing (CenterCrop <= 768, Resize 224x224) as Exp 7:
  - Phikon (owkin/phikon, HuggingFace open access, 768-dim ViT-B/16)
  - CTransPath (1aurent/swin_tiny_patch4_window7_224.CTransPath, 768-dim Swin-T)
  - UNI (optional via --include-uni if HF token has gated access)

For each outer fold:
  - StandardScaler + LogisticRegression(max_iter=2000, class_weight="balanced")
  - Selects C in {1e-3, 1e-2, 1e-1, 1, 10, 100} strictly by inner patient-grouped CV
  - Reports OOF accuracy, Wilson CI, cluster-bootstrap CI, balanced acc, macro F1,
    macro AUC, recall per class, accuracy by magnification, and per-image OOF probabilities.
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision.transforms as T
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

SRC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from common import (ImageList, MEAN, STD, metrics, cluster_bootstrap,
                    label_to_int, get_lunghist_root)
from train_scenarios import resolve_split_csv


def extract_phikon(df, root, dev, bs=32):
    from transformers import AutoImageProcessor, AutoModel
    model_id = "owkin/phikon"
    processor = AutoImageProcessor.from_pretrained(model_id)
    model = AutoModel.from_pretrained(model_id).to(dev).eval()

    mean = tuple(processor.image_mean) if hasattr(processor, "image_mean") else MEAN
    std = tuple(processor.image_std) if hasattr(processor, "image_std") else STD
    tf = T.Compose([
        T.Lambda(lambda im: T.CenterCrop(min(768, *im.size))(im)),
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean, std),
    ])
    dl = DataLoader(ImageList(df, tf, root), batch_size=bs, shuffle=False, num_workers=4, pin_memory=(dev == "cuda"))
    feats = []
    with torch.inference_mode():
        for x, _, _ in dl:
            out = model(pixel_values=x.to(dev))
            feats.append(out.last_hidden_state[:, 0, :].cpu().numpy())
    return np.concatenate(feats, axis=0)


def extract_ctranspath(df, root, dev, bs=32):
    import timm
    from huggingface_hub import hf_hub_download
    import safetensors.torch as st

    tf = T.Compose([
        T.Lambda(lambda im: T.CenterCrop(min(768, *im.size))(im)),
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(MEAN, STD),
    ])
    dl = DataLoader(ImageList(df, tf, root), batch_size=bs, shuffle=False, num_workers=4, pin_memory=(dev == "cuda"))

    def _build_ctranspath_modern_timm():
        m = timm.create_model("swin_tiny_patch4_window7_224", pretrained=False, num_classes=0)
        m.patch_embed.proj = nn.Sequential(
            nn.Conv2d(3, 12, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(12),
            nn.ReLU(),
            nn.Conv2d(12, 24, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(24),
            nn.ReLU(),
            nn.Conv2d(24, 96, kernel_size=1, stride=1, bias=True),
        )
        return m

    model = _build_ctranspath_modern_timm()
    try:
        sf_path = hf_hub_download(repo_id="1aurent/swin_tiny_patch4_window7_224.CTransPath", filename="model.safetensors")
        sd = st.load_file(sf_path)
    except Exception:
        ckpt_p = hf_hub_download(repo_id="jamesdolezal/CTransPath", filename="ctranspath.pth")
        raw = torch.load(ckpt_p, map_location="cpu")
        sd = raw["model"] if isinstance(raw, dict) and "model" in raw else raw

    remapped = {}
    for k, v in sd.items():
        if k.endswith("relative_position_index") or k.endswith("attn_mask") or k.startswith("head."):
            continue
        nk = k
        for idx in (0, 1, 2):
            if k.startswith(f"layers.{idx}.downsample."):
                nk = k.replace(f"layers.{idx}.downsample.", f"layers.{idx+1}.downsample.", 1)
                break
        remapped[nk] = v
    model.load_state_dict(remapped, strict=True)

    model = model.to(dev).eval()
    feats = []
    with torch.inference_mode():
        for x, _, _ in dl:
            feats.append(model(x.to(dev)).cpu().numpy())
    return np.concatenate(feats, axis=0)


def extract_uni(df, root, dev, bs=32):
    import timm
    model = timm.create_model("hf-hub:MahmoodLab/uni", pretrained=True, init_values=1e-5, dynamic_img_size=True).to(dev).eval()
    tf = T.Compose([
        T.Lambda(lambda im: T.CenterCrop(min(768, *im.size))(im)),
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(MEAN, STD),
    ])
    dl = DataLoader(ImageList(df, tf, root), batch_size=bs, shuffle=False, num_workers=4, pin_memory=(dev == "cuda"))
    feats = []
    with torch.inference_mode():
        for x, _, _ in dl:
            feats.append(model(x.to(dev)).cpu().numpy())
    return np.concatenate(feats, axis=0)


def run_linear_probe_cv(X, df, c_grid=(1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0), seed=42):
    y = df["label"].map(label_to_int).to_numpy()
    groups = df["patient"].to_numpy()
    folds = df["fold"].to_numpy()

    oof_probs = np.zeros((len(df), 3), dtype=np.float64)
    chosen_c = []

    for f in range(5):
        te_mask = (folds == f)
        tr_mask = ~te_mask
        X_tr, y_tr, g_tr = X[tr_mask], y[tr_mask], groups[tr_mask]
        X_te = X[te_mask]

        n_inner = min(5, len(np.unique(g_tr)))
        inner_cv = list(StratifiedGroupKFold(n_splits=n_inner, shuffle=True, random_state=seed).split(X_tr, y_tr, g_tr))

        best_c, best_score = c_grid[0], float("inf")
        for c_val in c_grid:
            fold_losses = []
            for itr, iva in inner_cv:
                clf = make_pipeline(
                    StandardScaler(),
                    LogisticRegression(C=c_val, max_iter=2000, class_weight="balanced", random_state=seed),
                )
                clf.fit(X_tr[itr], y_tr[itr])
                pr = clf.predict_proba(X_tr[iva])
                fold_losses.append(log_loss(y_tr[iva], pr, labels=[0, 1, 2]))
            mean_ll = float(np.mean(fold_losses))
            if mean_ll < best_score:
                best_score = mean_ll
                best_c = c_val

        chosen_c.append(float(best_c))
        final_clf = make_pipeline(
            StandardScaler(),
            LogisticRegression(C=best_c, max_iter=2000, class_weight="balanced", random_state=seed),
        )
        final_clf.fit(X_tr, y_tr)
        oof_probs[te_mask] = final_clf.predict_proba(X_te)

    pred = oof_probs.argmax(1)
    res = metrics(y, oof_probs)
    res["acc_cluster_ci"] = cluster_bootstrap(y, pred, groups)
    res["recall_per_class"] = [float(100.0 * (pred[y == k] == k).mean()) for k in range(3)]
    res["selected_C_per_fold"] = chosen_c
    if "magnification" in df.columns:
        res["acc_by_mag"] = {
            str(m): round(float(100.0 * (pred[df["magnification"] == m] == y[df["magnification"] == m]).mean()), 2)
            for m in df["magnification"].unique()
        }
    return oof_probs, res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="data_splits/lunghist700.csv")
    ap.add_argument("--folds", default="data_splits/folds_patient.csv")
    ap.add_argument("--root", default="")
    ap.add_argument("--models", default="phikon,ctranspath")
    ap.add_argument("--include-uni", action="store_true")
    ap.add_argument("--out", default="results_rev1/WP06/wp06d")
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    lh_root = get_lunghist_root(args.root)
    df = pd.read_csv(resolve_split_csv(args.csv))
    f_df = pd.read_csv(resolve_split_csv(args.folds))
    fold_map = dict(zip(f_df["path"], f_df["fold"]))
    df["fold"] = df["path"].map(fold_map).astype(int)

    model_list = [m.strip().lower() for m in args.models.split(",") if m.strip()]
    if args.include_uni and "uni" not in model_list:
        model_list.append("uni")

    extractors = {
        "phikon": extract_phikon,
        "ctranspath": extract_ctranspath,
        "uni": extract_uni,
    }

    summary = []
    for m_name in model_list:
        feat_path = os.path.join(args.out, f"wp06d_{m_name}_features.npz")
        try:
            if os.path.exists(feat_path):
                X = np.load(feat_path)["features"]
                print(f"[Cache] Loaded {m_name} features {X.shape}")
            else:
                print(f"[Extract] Extracting frozen {m_name} features on LungHist700 ({len(df)} imgs)...")
                X = extractors[m_name](df, lh_root, dev, args.bs)
                np.savez_compressed(feat_path, features=X)
                print(f"[Saved] {feat_path}: shape={X.shape}")
        except Exception as exc:
            print(f"[Warn] Skipping {m_name}: {exc}")
            continue

        oof_probs, res = run_linear_probe_cv(X, df, seed=args.seed)
        res["foundation_model"] = m_name
        res["feature_dim"] = int(X.shape[1])

        out_df = df.copy()
        out_df[["p_ACA", "p_N", "p_SCC"]] = oof_probs
        out_df["pred"] = oof_probs.argmax(1)
        out_df.to_csv(os.path.join(args.out, f"wp06d_{m_name}_oof.csv"), index=False)
        with open(os.path.join(args.out, f"wp06d_{m_name}_metrics.json"), "w", encoding="utf-8") as f:
            json.dump(res, f, indent=2)

        ci = res["acc_cluster_ci"]
        summary.append({
            "model": m_name,
            "dim": int(X.shape[1]),
            "oof_acc": round(res["acc"], 2),
            "cluster_ci_low": round(ci[0], 2),
            "cluster_ci_high": round(ci[1], 2),
            "bal_acc": round(res["bal_acc"], 2),
            "macro_f1": round(res["macro_f1"], 4),
            "macro_auc": round(res["macro_auc"], 4),
            "selected_C": res["selected_C_per_fold"],
        })
        print(f"[WP06d] {m_name:10s} | OOF Acc: {res['acc']:.2f}% [{ci[0]:.2f}-{ci[1]:.2f}%] | Macro F1: {res['macro_f1']:.4f} | AUC: {res['macro_auc']:.4f}")

    pd.DataFrame(summary).to_csv(os.path.join(args.out, "wp06d_summary.csv"), index=False)
    with open(os.path.join(args.out, "wp06d_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
