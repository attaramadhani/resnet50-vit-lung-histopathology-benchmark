"""WP01 - Shortcut test on LC25000 (Reviewer 5, comment 5.5b, "essential").

Question: can LC25000 be classified from shallow, non-morphological features?
If trivial classifiers reach >= 95% on the GROUP-AWARE test split, the near-ceiling
accuracy has a concrete shortcut explanation (acquisition batch, stain/colour, compression).

Feature sets (each fed to logistic regression and a random forest):
  F1 colour_hist   : 32-bin histogram per RGB channel + 32-bin per HSV channel (192 dims)
  F2 colour_moments: mean, std, skewness per RGB and per Lab channel (18 dims)
  F3 thumb16       : 16x16 RGB thumbnail, raw pixels (768 dims)
  F4 blur_thumb32  : Gaussian blur (sigma = 8 px at 256 px) then 32x32 RGB thumbnail (3072 dims)
  F5 gray_thumb32  : 32x32 grayscale thumbnail, no colour (1024 dims)
  F6 reinhard_thumb32: Reinhard stain/colour normalisation to one fixed target, then 32x32 RGB (3072 dims)

Splits evaluated (train on dev, test once on test):
  - group-aware split  : data_splits/exp4_grouped_split.csv (partition dev/test)   <- PRIMARY
  - image-level split  : data_splits/lc25000_dev.csv + lc25000_test.csv            <- secondary

Runs on CPU in about 15-30 minutes (feature extraction dominates). No GPU needed.

Usage (from the repository root):
  python wp01_shortcut_lc25000.py --root /workspace/data/LC25000/ --out results_wp01
  (--root must contain lung_aca/, lung_n/, lung_scc/ with the ORIGINAL 768x768 images,
   or the 256 px cache; both work.)

Outputs in --out:
  wp01_results.csv   one row per split x feature set x classifier (acc, Wilson CI, macro-F1, per-class recall)
  wp01_results.md    the same as a Markdown table, ready to paste into the manuscript
  wp01_features_*.npz cached features (delete to recompute)
"""
import argparse
import math
import os
import time

import numpy as np
import pandas as pd
from PIL import Image, ImageFilter
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, recall_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

CLASSES = ["ACA", "N", "SCC"]
ALIAS = {"LUNG_ACA": "ACA", "LUNG_N": "N", "LUNG_SCC": "SCC", "BENIGN": "N", "NORMAL": "N"}
SEED = 42


def lab(v):
    v = str(v).strip().upper()
    return CLASSES.index(ALIAS.get(v, v))


def wilson(k, n, z=1.959964):
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return 100 * (c - h), 100 * (c + h)


# ---------------------------------------------------------------- colour helpers (no extra deps)
def rgb_to_lab(rgb):
    """rgb float in [0,1], shape (...,3) -> CIE Lab (D65)."""
    a = np.where(rgb > 0.04045, ((rgb + 0.055) / 1.055) ** 2.4, rgb / 12.92)
    M = np.array([[0.4124, 0.3576, 0.1805], [0.2126, 0.7152, 0.0722], [0.0193, 0.1192, 0.9505]])
    xyz = a @ M.T / np.array([0.95047, 1.0, 1.08883])
    f = np.where(xyz > 0.008856, np.cbrt(xyz), 7.787 * xyz + 16 / 116)
    L = 116 * f[..., 1] - 16
    A = 500 * (f[..., 0] - f[..., 1])
    B = 200 * (f[..., 1] - f[..., 2])
    return np.stack([L, A, B], -1)


def lab_to_rgb(labi):
    L, A, B = labi[..., 0], labi[..., 1], labi[..., 2]
    fy = (L + 16) / 116
    fx = fy + A / 500
    fz = fy - B / 200
    f = np.stack([fx, fy, fz], -1)
    xyz = np.where(f ** 3 > 0.008856, f ** 3, (f - 16 / 116) / 7.787) * np.array([0.95047, 1.0, 1.08883])
    Minv = np.linalg.inv(np.array([[0.4124, 0.3576, 0.1805], [0.2126, 0.7152, 0.0722], [0.0193, 0.1192, 0.9505]]))
    a = xyz @ Minv.T
    rgb = np.where(a > 0.0031308, 1.055 * np.clip(a, 0, None) ** (1 / 2.4) - 0.055, 12.92 * a)
    return np.clip(rgb, 0, 1)


def skew(x, axis=0):
    m = x.mean(axis)
    s = x.std(axis) + 1e-8
    return (((x - m) / s) ** 3).mean(axis)


# ---------------------------------------------------------------- feature extraction
def load256(path):
    return Image.open(path).convert("RGB").resize((256, 256), Image.BILINEAR)


def features_for(im, target_stats):
    rgb = np.asarray(im, dtype=np.float32) / 255.0
    hsv = np.asarray(im.convert("HSV"), dtype=np.float32) / 255.0
    f1 = np.concatenate([np.histogram(rgb[..., c], 32, (0, 1))[0] for c in range(3)] +
                        [np.histogram(hsv[..., c], 32, (0, 1))[0] for c in range(3)]).astype(np.float32)
    f1 /= f1.sum() / 6.0
    px = rgb.reshape(-1, 3)
    lb = rgb_to_lab(rgb).reshape(-1, 3)
    f2 = np.concatenate([px.mean(0), px.std(0), skew(px), lb.mean(0), lb.std(0), skew(lb)]).astype(np.float32)
    f3 = (np.asarray(im.resize((16, 16), Image.BILINEAR), dtype=np.float32) / 255.0).ravel()
    f4 = (np.asarray(im.filter(ImageFilter.GaussianBlur(8)).resize((32, 32), Image.BILINEAR),
                     dtype=np.float32) / 255.0).ravel()
    f5 = (np.asarray(im.convert("L").resize((32, 32), Image.BILINEAR), dtype=np.float32) / 255.0).ravel()
    # Reinhard: match per-image Lab mean/std to a fixed target
    l_img = rgb_to_lab(rgb)
    mu, sd = l_img.reshape(-1, 3).mean(0), l_img.reshape(-1, 3).std(0) + 1e-6
    l_norm = (l_img - mu) / sd * target_stats[1] + target_stats[0]
    rn = Image.fromarray((lab_to_rgb(l_norm) * 255).astype(np.uint8))
    f6 = (np.asarray(rn.resize((32, 32), Image.BILINEAR), dtype=np.float32) / 255.0).ravel()
    return {"colour_hist": f1, "colour_moments": f2, "thumb16": f3, "blur_thumb32": f4,
            "gray_thumb32": f5, "reinhard_thumb32": f6}


def extract(df, root, target_stats, cache):
    if os.path.exists(cache):
        z = np.load(cache, allow_pickle=True)
        return {k: z[k] for k in z.files if k != "paths"}
    feats = {}
    t0 = time.time()
    for i, p in enumerate(df.path):
        f = features_for(load256(os.path.join(root, p)), target_stats)
        for k, v in f.items():
            feats.setdefault(k, []).append(v)
        if (i + 1) % 1000 == 0:
            print(f"  {i + 1}/{len(df)} images ({time.time() - t0:.0f}s)")
    feats = {k: np.stack(v) for k, v in feats.items()}
    np.savez_compressed(cache, paths=df.path.values, **feats)
    return feats


def classifiers():
    return {
        "logreg": make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, C=1.0, random_state=SEED)),
        "random_forest": RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=SEED),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="folder with lung_aca/, lung_n/, lung_scc/")
    ap.add_argument("--splits", default="data_splits", help="folder with the split CSVs of v2.0.0")
    ap.add_argument("--out", default="results_wp01")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    allimg = pd.read_csv(os.path.join(a.splits, "lc25000_lung_all.csv"))
    # fixed Reinhard target = mean Lab statistics of 300 random images (seed 42)
    rs = allimg.sample(300, random_state=SEED)
    stats = [rgb_to_lab(np.asarray(load256(os.path.join(a.root, p)), dtype=np.float32) / 255.0).reshape(-1, 3)
             for p in rs.path]
    target = (np.mean([s.mean(0) for s in stats], 0), np.mean([s.std(0) for s in stats], 0))

    print("Extracting features for all 15,000 images (cached after the first run)...")
    F = extract(allimg, a.root, target, os.path.join(a.out, "wp01_features_all.npz"))
    idx = {p: i for i, p in enumerate(allimg.path)}

    grouped = pd.read_csv(os.path.join(a.splits, "exp4_grouped_split.csv"))
    dev_i = pd.read_csv(os.path.join(a.splits, "lc25000_dev.csv"))
    test_i = pd.read_csv(os.path.join(a.splits, "lc25000_test.csv"))
    splits = {
        "group-aware": (grouped[grouped.partition == "dev"], grouped[grouped.partition == "test"]),
        "image-level": (dev_i, test_i),
    }

    rows = []
    for sname, (tr, te) in splits.items():
        itr = np.array([idx[p] for p in tr.path])
        ite = np.array([idx[p] for p in te.path])
        ytr = np.array([lab(v) for v in tr.label])
        yte = np.array([lab(v) for v in te.label])
        for fname, X in F.items():
            for cname, clf in classifiers().items():
                clf.fit(X[itr], ytr)
                pred = clf.predict(X[ite])
                k = int((pred == yte).sum())
                lo, hi = wilson(k, len(yte))
                rec = recall_score(yte, pred, average=None, labels=[0, 1, 2])
                rows.append(dict(split=sname, features=fname, classifier=cname, n_train=len(itr), n_test=len(ite),
                                 correct=k, acc=round(100 * accuracy_score(yte, pred), 2),
                                 acc_ci_low=round(lo, 2), acc_ci_high=round(hi, 2),
                                 macro_f1=round(f1_score(yte, pred, average="macro"), 4),
                                 recall_ACA=round(100 * rec[0], 1), recall_N=round(100 * rec[1], 1),
                                 recall_SCC=round(100 * rec[2], 1)))
                print(f"[{sname:11s}] {fname:17s} {cname:13s} acc = {rows[-1]['acc']:6.2f}%")

    R = pd.DataFrame(rows)
    R.to_csv(os.path.join(a.out, "wp01_results.csv"), index=False)
    with open(os.path.join(a.out, "wp01_results.md"), "w") as f:
        f.write("| Split | Features | Classifier | Test acc. (%) [95% Wilson CI] | Macro F1 | Recall ACA / N / SCC (%) |\n")
        f.write("| --- | --- | --- | --- | --- | --- |\n")
        for r in rows:
            f.write(f"| {r['split']} | {r['features']} | {r['classifier']} | {r['acc']:.2f} [{r['acc_ci_low']:.1f}-"
                    f"{r['acc_ci_high']:.1f}] | {r['macro_f1']:.4f} | {r['recall_ACA']} / {r['recall_N']} / {r['recall_SCC']} |\n")
    best = R[R.split == "group-aware"].sort_values("acc", ascending=False).iloc[0]
    print(f"\nBest shallow classifier on the GROUP-AWARE split: {best.features} + {best.classifier} = {best.acc:.2f}%")
    print("Interpretation rule (agreed in advance): >= 95% -> LC25000 is shortcut-solvable; "
          "< 80% -> shortcut explanation unlikely; in between -> partial shortcut.")


if __name__ == "__main__":
    main()
