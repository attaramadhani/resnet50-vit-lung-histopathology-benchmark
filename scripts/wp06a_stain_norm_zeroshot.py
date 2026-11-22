"""WP06(a) - Stain normalization (Macenko & Reinhard) for zero-shot LungHist700 ensembles.

Evaluates the existing LC25000 main ensembles (Scenario 1 ResNet50 & Scenario 6 ViT-S/16,
without re-training) on all 691 LungHist700 images after stain normalization using
one deterministic representative LC25000 image (seed=42) as reference target.

Matrix: {macenko, reinhard} x {resize, crop768} x {resnet50, vit_s16} = 8 zero-shot runs.
"""
import argparse
import json
import os
import sys

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T

SRC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from common import (build_model, load_checkpoint, resolve_checkpoint_path, metrics, cluster_bootstrap,
                    label_to_int, MEAN, STD, get_lc25000_root, get_lunghist_root, resolve_split_csv)


class ReinhardNormalizer:
    """Reinhard et al. (2001) CIE Lab color transfer."""

    def __init__(self, target_rgb):
        lab = cv2.cvtColor(target_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
        self.t_mean = lab.reshape(-1, 3).mean(axis=0)
        self.t_std = np.maximum(lab.reshape(-1, 3).std(axis=0), 1e-6)

    def __call__(self, src_rgb):
        lab = cv2.cvtColor(src_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
        s_mean = lab.reshape(-1, 3).mean(axis=0)
        s_std = np.maximum(lab.reshape(-1, 3).std(axis=0), 1e-6)
        norm = (lab - s_mean) / s_std * self.t_std + self.t_mean
        norm = np.clip(norm, 0, 255).astype(np.uint8)
        return cv2.cvtColor(norm, cv2.COLOR_LAB2RGB)


class MacenkoNormalizer:
    """Macenko et al. (2009) optical density SVD stain normalization (self-contained NumPy)."""

    def __init__(self, target_rgb, io=240.0, alpha=1.0, beta=0.15):
        self.io = io
        self.alpha = alpha
        self.beta = beta
        self.he_ref, self.maxc_ref = self._fit(target_rgb)

    def _rgb_to_od(self, rgb):
        img = rgb.astype(np.float32).reshape(-1, 3)
        img = np.clip(img, 1.0, 255.0)
        return -np.log(img / self.io)

    def _get_stain_matrix(self, od):
        od_hat = od[np.all(od > self.beta, axis=1)]
        if len(od_hat) < 32:
            # Fallback to Ruifrok H&E basis if patch is mostly glass
            return np.array([[0.5626, 0.2159], [0.7201, 0.8012], [0.4062, 0.5581]], dtype=np.float32)
        _, eigvecs = np.linalg.eigh(np.cov(od_hat.T))
        that = od_hat @ eigvecs[:, 1:3]
        phi = np.arctan2(that[:, 1], that[:, 0])
        min_phi = np.percentile(phi, self.alpha)
        max_phi = np.percentile(phi, 100.0 - self.alpha)
        v_min = eigvecs[:, 1:3] @ np.array([np.cos(min_phi), np.sin(min_phi)])
        v_max = eigvecs[:, 1:3] @ np.array([np.cos(max_phi), np.sin(max_phi)])
        if v_min[0] > v_max[0]:
            he = np.stack([v_min, v_max], axis=1)
        else:
            he = np.stack([v_max, v_min], axis=1)
        he /= np.maximum(np.linalg.norm(he, axis=0, keepdims=True), 1e-8)
        return he.astype(np.float32)

    def _fit(self, rgb):
        od = self._rgb_to_od(rgb)
        he = self._get_stain_matrix(od)
        c, _, _, _ = np.linalg.lstsq(he, od.T, rcond=None)
        maxc = np.percentile(c, 99.0, axis=1, keepdims=True)
        return he, np.maximum(maxc, 1e-4)

    def __call__(self, src_rgb):
        h, w, _ = src_rgb.shape
        od = self._rgb_to_od(src_rgb)
        try:
            he_src = self._get_stain_matrix(od)
            c, _, _, _ = np.linalg.lstsq(he_src, od.T, rcond=None)
            maxc_src = np.maximum(np.percentile(c, 99.0, axis=1, keepdims=True), 1e-4)
            c_norm = c * (self.maxc_ref / maxc_src)
            od_norm = self.he_ref @ c_norm
            rgb_norm = self.io * np.exp(-od_norm)
            rgb_norm = np.clip(rgb_norm.T.reshape(h, w, 3), 0, 255).astype(np.uint8)
            return rgb_norm
        except np.linalg.LinAlgError:
            return src_rgb


class StainNormDataset(Dataset):
    def __init__(self, df, root, preprocess, normalizer):
        self.df = df.reset_index(drop=True)
        self.root = root
        self.preprocess = preprocess
        self.normalizer = normalizer
        self.to_tensor = T.Compose([T.ToTensor(), T.Normalize(MEAN, STD)])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        rel = str(r["path"]).replace("\\", "/").lstrip("/")
        p = os.path.join(self.root, rel)
        im = Image.open(p).convert("RGB")
        if self.preprocess == "crop768":
            s = min(768, im.width, im.height)
            im = T.CenterCrop(s)(im)
        im = im.resize((224, 224), Image.BILINEAR)
        arr = self.normalizer(np.asarray(im, dtype=np.uint8))
        return self.to_tensor(Image.fromarray(arr)), label_to_int(r["label"]), i


def select_target_lc25000(lc_root, dev_csv, seed=42):
    df = pd.read_csv(dev_csv)
    rng = np.random.default_rng(seed)
    idx = int(rng.integers(0, len(df)))
    row = df.iloc[idx]
    rel = str(row["path"]).replace("\\", "/").lstrip("/")
    full_path = os.path.join(lc_root, rel)
    arr = np.asarray(Image.open(full_path).convert("RGB").resize((256, 256), Image.BILINEAR), dtype=np.uint8)
    return arr, {"index": idx, "path": row["path"], "label": row["label"], "resolved_path": full_path, "seed": seed}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lh-csv", default="data_splits/lunghist700.csv")
    ap.add_argument("--lc-dev-csv", default="data_splits/lc25000_dev.csv")
    ap.add_argument("--lh-root", default="")
    ap.add_argument("--lc-root", default="")
    ap.add_argument("--ckpt-root", default="old_ckpts/runs")
    ap.add_argument("--out", default="results_rev1/WP06/wp06a")
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    lh_root = get_lunghist_root(args.lh_root)
    lc_root = get_lc25000_root(args.lc_root)
    lh_csv = resolve_split_csv(args.lh_csv)
    lc_dev_csv = resolve_split_csv(args.lc_dev_csv)

    ckpt_base = args.ckpt_root if os.path.isdir(args.ckpt_root) else "runs"
    target_arr, target_meta = select_target_lc25000(lc_root, lc_dev_csv, args.seed)
    with open(os.path.join(args.out, "wp06a_target_ref.json"), "w", encoding="utf-8") as f:
        json.dump(target_meta, f, indent=2)
    print(f"[Target] LC25000 reference image (seed={args.seed}): {target_meta['path']}")

    normalizers = {
        "macenko": MacenkoNormalizer(target_arr),
        "reinhard": ReinhardNormalizer(target_arr),
    }

    lh_df = pd.read_csv(lh_csv)
    y = lh_df["label"].map(label_to_int).to_numpy()
    summary_rows = []

    for norm_name, normalizer in normalizers.items():
        for preproc in ["resize", "crop768"]:
            ds = StainNormDataset(lh_df, lh_root, preproc, normalizer)
            dl = DataLoader(ds, batch_size=args.bs, shuffle=False, num_workers=4, pin_memory=(dev == "cuda"))

            for arch, folder in [("resnet50", "resnet50_main"), ("vit_s16", "vit_s_main")]:
                ckpts = [resolve_checkpoint_path(os.path.join(ckpt_base, folder, f"fold{k}.pt")) for k in range(5)]
                probs = []
                for c in ckpts:
                    m = load_checkpoint(build_model(arch, num_classes=3, pretrained=False), c, dev).to(dev).eval()
                    p_list = []
                    with torch.no_grad(), torch.autocast(dev, dtype=torch.float16, enabled=(dev == "cuda")):
                        for x, _, _ in dl:
                            p_list.append(torch.softmax(m(x.to(dev)).float(), 1).cpu().numpy())
                    probs.append(np.concatenate(p_list, axis=0))

                ens_prob = np.mean(probs, axis=0)
                pred = ens_prob.argmax(1)
                res = metrics(y, ens_prob)
                res["recall_per_class"] = [float(100.0 * (pred[y == k] == k).mean()) for k in range(3)]
                if "patient" in lh_df.columns:
                    res["acc_cluster_ci"] = cluster_bootstrap(y, pred, lh_df["patient"].to_numpy())
                if "magnification" in lh_df.columns:
                    res["acc_by_mag"] = {
                        str(g): round(float(100.0 * (pred[lh_df.magnification == g] == y[lh_df.magnification == g]).mean()), 2)
                        for g in lh_df.magnification.unique()
                    }
                res["stain_norm"] = norm_name
                res["preprocess"] = preproc
                res["arch"] = arch
                res["target_ref"] = target_meta["path"]

                tag = f"wp06a_{norm_name}_{preproc}_{arch}"
                out_df = lh_df.copy()
                out_df[["p_ACA", "p_N", "p_SCC"]] = ens_prob
                out_df["pred"] = pred
                out_df.to_csv(os.path.join(args.out, f"{tag}_preds.csv"), index=False)
                with open(os.path.join(args.out, f"{tag}_metrics.json"), "w", encoding="utf-8") as f:
                    json.dump(res, f, indent=2)

                ci = res.get("acc_cluster_ci", res["acc_ci"])
                summary_rows.append({
                    "norm": norm_name,
                    "preprocess": preproc,
                    "arch": arch,
                    "acc": round(res["acc"], 2),
                    "cluster_ci_low": round(ci[0], 2),
                    "cluster_ci_high": round(ci[1], 2),
                    "bal_acc": round(res["bal_acc"], 2),
                    "macro_f1": round(res["macro_f1"], 4),
                    "macro_auc": round(res["macro_auc"], 4),
                })
                print(f"[WP06a] {norm_name:8s} | {preproc:7s} | {arch:8s} -> Acc: {res['acc']:.2f}% [{ci[0]:.2f}-{ci[1]:.2f}%] | AUC: {res['macro_auc']:.4f}")

    pd.DataFrame(summary_rows).to_csv(os.path.join(args.out, "wp06a_summary.csv"), index=False)
    with open(os.path.join(args.out, "wp06a_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary_rows, f, indent=2)


if __name__ == "__main__":
    main()
