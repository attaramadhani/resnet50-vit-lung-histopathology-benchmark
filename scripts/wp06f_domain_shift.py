"""WP06(f) - Domain Shift Analysis between LC25000 (768x768) and LungHist700.

Samples 300 random images per dataset (seed=42) and computes:
  - Macenko stain matrix vectors (H & E optical density vectors)
  - Ruifrok-Johnston HED optical density channel histograms
  - ResNet50 (ImageNet-1K V2) penultimate feature MMD^2 (RBF median heuristic) & FID
  - 2D UMAP projection plot (300 DPI) showing domain shift between datasets
"""
import argparse
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T
from PIL import Image
from scipy import linalg
from torch.utils.data import DataLoader

SRC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from common import MEAN, STD, ImageList, get_lc25000_root, get_lunghist_root, resolve_split_csv


RGB_FROM_HED = np.array([
    [0.65, 0.70, 0.29],
    [0.07, 0.99, 0.11],
    [0.27, 0.57, 0.78],
], dtype=np.float32)
RGB_FROM_HED /= np.linalg.norm(RGB_FROM_HED, axis=1, keepdims=True)
HED_FROM_RGB = np.linalg.inv(RGB_FROM_HED)


def macenko_he_vectors(rgb, io=240.0, alpha=1.0, beta=0.15):
    od = -np.log(np.clip(rgb.astype(np.float32).reshape(-1, 3), 1.0, 255.0) / io)
    od_hat = od[np.all(od > beta, axis=1)]
    if len(od_hat) < 32:
        return np.array([0.5626, 0.7201, 0.4062]), np.array([0.2159, 0.8012, 0.5581])
    _, eigvecs = np.linalg.eigh(np.cov(od_hat.T))
    that = od_hat @ eigvecs[:, 1:3]
    phi = np.arctan2(that[:, 1], that[:, 0])
    v_min = eigvecs[:, 1:3] @ np.array([np.cos(np.percentile(phi, alpha)), np.sin(np.percentile(phi, alpha))])
    v_max = eigvecs[:, 1:3] @ np.array([np.cos(np.percentile(phi, 100.0 - alpha)), np.sin(np.percentile(phi, 100.0 - alpha))])
    v1, v2 = (v_min, v_max) if v_min[0] > v_max[0] else (v_max, v_min)
    v1 = v1 / max(np.linalg.norm(v1), 1e-8)
    v2 = v2 / max(np.linalg.norm(v2), 1e-8)
    return v1, v2


def compute_hed_stats(df, root):
    h_vecs, e_vecs = [], []
    hed_hists = np.zeros((3, 50), dtype=np.float64)
    edges = np.linspace(0.0, 1.5, 51)
    for r in df.itertuples():
        rel = str(r.path).replace("\\", "/").lstrip("/")
        im = Image.open(os.path.join(root, rel)).convert("RGB")
        # Center crop <=768 and resize to 256x256 for comparable OD stats
        s = min(768, im.width, im.height)
        im = T.CenterCrop(s)(im).resize((256, 256), Image.BILINEAR)
        arr = np.asarray(im, dtype=np.uint8)
        hv, ev = macenko_he_vectors(arr)
        h_vecs.append(hv)
        e_vecs.append(ev)

        od = -np.log10(np.clip(arr.astype(np.float32) / 255.0, 1e-6, 1.0))
        hed = np.clip(od @ HED_FROM_RGB, 0.0, 1.5)
        for ch in range(3):
            cnt, _ = np.histogram(hed[:, :, ch], bins=edges)
            hed_hists[ch] += cnt
    hed_hists = hed_hists / np.maximum(hed_hists.sum(axis=1, keepdims=True), 1.0)
    return {
        "macenko_H_mean": np.mean(h_vecs, axis=0).round(4).tolist(),
        "macenko_H_std": np.std(h_vecs, axis=0).round(4).tolist(),
        "macenko_E_mean": np.mean(e_vecs, axis=0).round(4).tolist(),
        "macenko_E_std": np.std(e_vecs, axis=0).round(4).tolist(),
        "hed_hist_bins": edges.round(4).tolist(),
        "hed_hist_density": hed_hists.round(6).tolist(),
    }


def extract_r50_features(df, root, dev, bs=32):
    enc = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V2)
    enc.fc = nn.Identity()
    enc = enc.to(dev).eval()
    tf = T.Compose([
        T.Lambda(lambda im: T.CenterCrop(min(768, *im.size))(im)),
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(MEAN, STD),
    ])
    dl = DataLoader(ImageList(df, tf, root), batch_size=bs, shuffle=False, num_workers=4)
    feats = []
    with torch.inference_mode():
        for x, _, _ in dl:
            feats.append(enc(x.to(dev)).cpu().numpy())
    return np.concatenate(feats, axis=0)


def compute_fid(f1, f2):
    mu1, sig1 = f1.mean(axis=0), np.cov(f1, rowvar=False)
    mu2, sig2 = f2.mean(axis=0), np.cov(f2, rowvar=False)
    diff = mu1 - mu2
    eps = 1e-6 * np.eye(sig1.shape[0], dtype=np.float64)
    res_sqrt = linalg.sqrtm((sig1 + eps) @ (sig2 + eps))
    covmean = res_sqrt[0] if isinstance(res_sqrt, tuple) else res_sqrt
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return float(diff @ diff + np.trace(sig1 + sig2 - 2.0 * covmean))


def compute_mmd_rbf(X, Y):
    Z = np.vstack([X, Y])
    d2 = np.sum((Z[:, None, :] - Z[None, :, :]) ** 2, axis=-1)
    med = np.median(d2[d2 > 0])
    gamma = 1.0 / max(med, 1e-8)

    def rbf(A, B):
        dist2 = np.sum((A[:, None, :] - B[None, :, :]) ** 2, axis=-1)
        return np.exp(-gamma * dist2)

    Kxx = rbf(X, X)
    Kyy = rbf(Y, Y)
    Kxy = rbf(X, Y)
    n, m = len(X), len(Y)
    np.fill_diagonal(Kxx, 0.0)
    np.fill_diagonal(Kyy, 0.0)
    mmd2 = Kxx.sum() / (n * (n - 1)) + Kyy.sum() / (m * (m - 1)) - 2.0 * Kxy.mean()
    return float(mmd2), float(gamma)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lc-csv", default="data_splits/lc25000_lung_all.csv")
    ap.add_argument("--lh-csv", default="data_splits/lunghist700.csv")
    ap.add_argument("--lc-raw-root", default="", help="Original 768x768 LC25000 root (auto-resolved if empty)")
    ap.add_argument("--lh-root", default="")
    ap.add_argument("--n-per-dataset", type=int, default=300)
    ap.add_argument("--out", default="results_rev1/WP06/wp06f")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    lh_root = get_lunghist_root(args.lh_root)
    lc_raw_root = get_lc25000_root(args.lc_raw_root, require_raw=True)
    if not lc_raw_root or not os.path.isdir(os.path.join(lc_raw_root, "lung_aca")):
        raise FileNotFoundError(f"Original 768x768 LC25000 root not found: '{lc_raw_root}'")

    lc_df = pd.read_csv(resolve_split_csv(args.lc_csv)).sample(n=args.n_per_dataset, random_state=args.seed).reset_index(drop=True)
    lh_df = pd.read_csv(resolve_split_csv(args.lh_csv)).sample(n=args.n_per_dataset, random_state=args.seed).reset_index(drop=True)
    lc_df["dataset"] = "LC25000"
    lh_df["dataset"] = "LungHist700"
    pd.concat([lc_df[["path", "label", "dataset"]], lh_df[["path", "label", "dataset"]]], ignore_index=True).to_csv(
        os.path.join(args.out, "wp06f_sampled_images.csv"), index=False
    )

    print("[WP06f] Computing Macenko stain vectors & HED histograms...")
    lc_stain = compute_hed_stats(lc_df, lc_raw_root)
    lh_stain = compute_hed_stats(lh_df, lh_root)

    print("[WP06f] Extracting ResNet50 ImageNet features for MMD, FID & UMAP...")
    f_lc = extract_r50_features(lc_df, lc_raw_root, dev)
    f_lh = extract_r50_features(lh_df, lh_root, dev)

    mmd2, gamma = compute_mmd_rbf(f_lc, f_lh)
    fid = compute_fid(f_lc, f_lh)

    import umap
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=args.seed)
    emb = reducer.fit_transform(np.vstack([f_lc, f_lh]))

    # Plot 1: UMAP
    fig, ax = plt.subplots(figsize=(7.2, 5.6), dpi=300)
    colors = {"ACA": "#D55E00", "N": "#009E73", "SCC": "#0072B2"}
    n = len(lc_df)
    for cls in ["ACA", "N", "SCC"]:
        m_lc = (lc_df["label"] == cls).to_numpy()
        m_lh = (lh_df["label"] == cls).to_numpy()
        ax.scatter(emb[:n][m_lc, 0], emb[:n][m_lc, 1], c=colors[cls], marker="o", s=22, alpha=0.65, label=f"LC25000 - {cls}")
        ax.scatter(emb[n:][m_lh, 0], emb[n:][m_lh, 1], c=colors[cls], marker="^", s=34, alpha=0.80, edgecolors="black", linewidths=0.4, label=f"LungHist700 - {cls}")
    ax.set_title(f"ResNet50 Feature UMAP (n=300/dataset) | MMD^2={mmd2:.4f}, FID={fid:.2f}", fontsize=10.5)
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    ax.legend(frameon=True, fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "wp06f_umap_domain_shift.png"), dpi=300)
    plt.close(fig)

    # Plot 2: HED histograms
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.6), dpi=300)
    bins = np.array(lc_stain["hed_hist_bins"])
    centers = 0.5 * (bins[:-1] + bins[1:])
    for i, ch_name in enumerate(["Hematoxylin (H)", "Eosin (E)", "DAB Residual (D)"]):
        axes[i].plot(centers, lc_stain["hed_hist_density"][i], lw=2, label="LC25000 (768px)", color="#0072B2")
        axes[i].plot(centers, lh_stain["hed_hist_density"][i], lw=2, label="LungHist700", color="#D55E00")
        axes[i].set_title(ch_name, fontsize=10)
        axes[i].set_xlabel("Optical Density")
        axes[i].set_ylabel("Density")
        axes[i].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "wp06f_hed_histograms.png"), dpi=300)
    plt.close(fig)

    res = {
        "n_per_dataset": args.n_per_dataset,
        "seed": args.seed,
        "mmd2_rbf": round(mmd2, 6),
        "mmd_gamma": float(gamma),
        "fid_resnet50": round(fid, 3),
        "lc25000_stain": lc_stain,
        "lunghist700_stain": lh_stain,
    }
    with open(os.path.join(args.out, "wp06f_domain_shift_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2)
    print(f"[OK] WP06f Domain Shift: MMD^2={mmd2:.6f} | FID={fid:.2f}")


if __name__ == "__main__":
    main()
