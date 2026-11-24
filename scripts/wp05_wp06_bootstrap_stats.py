"""wp05_wp06_bootstrap_stats.py - Compute Wilson CI, cluster-bootstrap CI, ECE, Brier, and per-class sens/spec.

Generates Statistik_CI_Kalibrasi_SensSpec.csv directly from per-image prediction CSVs in results_rev1/.
"""
import argparse
import math
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, f1_score, balanced_accuracy_score

LMAP = {"ACA": 0, "N": 1, "SCC": 2}
Z95 = 1.959963984540054


def wilson_ci(k: int, n: int):
    p = k / n
    denom = 1.0 + Z95**2 / n
    c = (p + Z95**2 / (2 * n)) / denom
    h = (Z95 * math.sqrt((p * (1 - p) + Z95**2 / (4 * n)) / n)) / denom
    return round(100 * (c - h), 2), round(100 * (c + h), 2)


def compute_ece_brier(y_true: np.ndarray, probs: np.ndarray, n_bins: int = 15):
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    acc = (pred == y_true).astype(float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y_true)
    for b0, b1 in zip(bins[:-1], bins[1:]):
        mask = (conf > b0) & (conf <= b1) if b0 > 0 else (conf >= b0) & (conf <= b1)
        if mask.any():
            ece += (mask.sum() / n) * abs(acc[mask].mean() - conf[mask].mean())
    onehot = np.eye(3)[y_true]
    brier = float(np.mean(np.sum((probs - onehot) ** 2, axis=1)))
    return round(float(ece), 4), round(brier, 4)


def cluster_bootstrap_ci(y_true: np.ndarray, y_pred: np.ndarray, clusters: np.ndarray, n_boot: int = 2000, seed: int = 42):
    rng = np.random.default_rng(seed)
    uniq = np.unique(clusters)
    idx_by_cluster = [np.where(clusters == c)[0] for c in uniq]
    correct = (y_true == y_pred).astype(float)
    c_sums = np.array([correct[idx].sum() for idx in idx_by_cluster])
    c_lens = np.array([len(idx) for idx in idx_by_cluster])
    k_clusters = len(uniq)
    boots = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        draw = rng.integers(0, k_clusters, size=k_clusters)
        boots[b] = 100.0 * c_sums[draw].sum() / c_lens[draw].sum()
    return round(float(np.percentile(boots, 2.5)), 2), round(float(np.percentile(boots, 97.5)), 2)


def eval_preds_csv(wp: str, model_name: str, csv_path: Path, cluster_col: str, n_boot: int = 2000, seed: int = 42):
    df = pd.read_csv(csv_path)
    y_true = df["label"].map(LMAP).astype(int).to_numpy()
    probs = df[["p_ACA", "p_N", "p_SCC"]].to_numpy(dtype=float)
    y_pred = df["pred"].astype(int).to_numpy()
    n = len(df)
    k = int((y_true == y_pred).sum())
    acc = 100.0 * k / n
    w_lo, w_hi = wilson_ci(k, n)
    clusters = df[cluster_col].to_numpy()
    b_lo, b_hi = cluster_bootstrap_ci(y_true, y_pred, clusters, n_boot=n_boot, seed=seed)
    ece, brier = compute_ece_brier(y_true, probs)
    bal_acc = 100.0 * balanced_accuracy_score(y_true, y_pred)
    mf1 = f1_score(y_true, y_pred, average="macro")
    mauc = roc_auc_score(y_true, probs, multi_class="ovr", average="macro")

    sens, spec = [], []
    for c in range(3):
        tp = int(((y_true == c) & (y_pred == c)).sum())
        fn = int(((y_true == c) & (y_pred != c)).sum())
        tn = int(((y_true != c) & (y_pred != c)).sum())
        fp = int(((y_true != c) & (y_pred == c)).sum())
        sens.append(round(100.0 * tp / max(1, tp + fn), 2))
        spec.append(round(100.0 * tn / max(1, tn + fp), 2))

    return {
        "WP": wp,
        "Model": model_name,
        "n": n,
        "n_clusters": int(pd.Series(clusters).nunique()),
        "Acc (%)": round(acc, 2),
        "Wilson 95% CI": f"[{w_lo:.2f}, {w_hi:.2f}]",
        "Cluster Boot 95% CI": f"[{b_lo:.2f}, {b_hi:.2f}]",
        "Bal Acc (%)": round(bal_acc, 2),
        "Macro F1": round(mf1, 4),
        "Macro AUC": round(mauc, 4),
        "ECE (15-bin)": ece,
        "Brier": brier,
        "Sens ACA/N/SCC (%)": f"{sens[0]} / {sens[1]} / {sens[2]}",
        "Spec ACA/N/SCC (%)": f"{spec[0]} / {spec[1]} / {spec[2]}",
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-root", default="results_rev1")
    ap.add_argument("--out", default="results_rev1/Statistik_CI_Kalibrasi_SensSpec.csv")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    res = Path(args.results_root)
    rows = []
    for s in [42, 1, 2]:
        for arch, lbl in [("resnet50", "ResNet50"), ("vit_s16", "ViT-S/16")]:
            p = res / "WP03" / "group_aware" / f"seed_{s}" / f"wp03_grp_{arch}" / "ensemble_test_preds.csv"
            if p.exists():
                rows.append(eval_preds_csv("WP03_group_aware", f"{lbl}_seed{s}", p, "group", args.n_boot, args.seed))

    for fn, lbl in [("G1_clean_resnet50_ensemble_test_preds.csv", "G1_clean_resnet50_seed42"), ("G2_clean_vit_s16_ensemble_test_preds.csv", "G2_clean_vit_s16_seed42")]:
        p = res / "WP05" / fn
        if p.exists():
            rows.append(eval_preds_csv("WP05", lbl, p, "group", args.n_boot, args.seed))

    for norm in ["macenko", "reinhard"]:
        for prep in ["resize", "crop768"]:
            for arch in ["resnet50", "vit_s16"]:
                p = res / "WP06" / "wp06a" / f"wp06a_{norm}_{prep}_{arch}_preds.csv"
                if p.exists():
                    rows.append(eval_preds_csv("WP06a_ZeroShot", f"{norm}_{prep}_{arch}", p, "patient", args.n_boot, args.seed))

    for sub, fn, lbl in [
        ("wp06b", "e7_r50_hed_oof.csv", "WP06b_ResNet50_HED"),
        ("wp06b", "e7_r50_lcinit_hed_oof.csv", "WP06b_ResNet50_LCInit_HED"),
        ("wp06b", "e7_vit_hed_oof.csv", "WP06b_ViTS16_HED"),
        ("wp06c", "e7_vit_lcinit_oof.csv", "WP06c_ViTS16_LCInit"),
        ("wp06d", "wp06d_phikon_oof.csv", "WP06d_Phikon_LinProbe"),
        ("wp06d", "wp06d_ctranspath_oof.csv", "WP06d_CTransPath_LinProbe"),
        ("wp06e", "wp06e_tile_resnet50_oof.csv", "WP06e_Tile_ResNet50"),
        ("wp06e", "wp06e_tile_vit_s16_oof.csv", "WP06e_Tile_ViTS16"),
    ]:
        p = res / "WP06" / sub / fn
        if p.exists():
            rows.append(eval_preds_csv("WP06_LungHist700", lbl, p, "patient", args.n_boot, args.seed))

    out_df = pd.DataFrame(rows)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.out, index=False)
    print(f"[OK] Saved {len(out_df)} rows to {args.out}")


if __name__ == "__main__":
    main()
