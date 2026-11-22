"""Clustered McNemar (Durkalski et al., 2003) and Exact McNemar for LungHist700 (Section 3.7).

Evaluates the 3 model pairs on the patient-grouped LungHist700 predictions:
  Pair 1: ResNet50 (ImageNet) vs ViT-S/16 (ImageNet)             [e7_r50 vs e7_vit]
  Pair 2: ResNet50 (ImageNet) vs ResNet50 (LC25000 init)        [e7_r50 vs e7_r50_lcinit]
  Pair 3: ResNet50 (LC25000 init) vs ViT-S/16 (ImageNet)        [e7_r50_lcinit vs e7_vit]

With Holm-Bonferroni step-down correction for multiple testing across the 3 pairs.
Exact McNemar (binomial test) is reported as reference.

Reference:
  Durkalski, V. L., Evans, G. W., Palesch, Y. Y., & Lipsitz, S. R. (2003).
  Analysis of clustered matched-pair data. Statistics in Medicine, 22(14), 2307-2317.
"""
import argparse
import json
import os
import numpy as np
import pandas as pd
from scipy import stats


def clustered_mcnemar_durkalski(y_true, pred1, pred2, clusters):
    """Computes Durkalski clustered McNemar test statistic and p-value.

    Parameters:
      y_true: array-like of true class labels (N,)
      pred1: array-like of model 1 predictions (N,)
      pred2: array-like of model 2 predictions (N,)
      clusters: array-like of cluster IDs (e.g. patient ID) (N,)

    Returns:
      dict with test statistic, p-value, n10, n01, n_clusters, and exact McNemar ref
    """
    y_true = np.asarray(y_true)
    pred1 = np.asarray(pred1)
    pred2 = np.asarray(pred2)
    clusters = np.asarray(clusters)

    correct1 = (pred1 == y_true)
    correct2 = (pred2 == y_true)

    # Discordant indicators per observation
    # 10: Model 1 correct, Model 2 incorrect
    # 01: Model 1 incorrect, Model 2 correct
    ind_10 = correct1 & (~correct2)
    ind_01 = (~correct1) & correct2

    unique_clusters = np.unique(clusters)
    M = len(unique_clusters)

    d_per_cluster = []
    n10_total = int(ind_10.sum())
    n01_total = int(ind_01.sum())

    for cl in unique_clusters:
        mask = (clusters == cl)
        c_10 = ind_10[mask].sum()
        c_01 = ind_01[mask].sum()
        d_per_cluster.append(c_10 - c_01)

    d_arr = np.array(d_per_cluster, dtype=float)
    sum_d = d_arr.sum()
    sum_d_sq = (d_arr ** 2).sum()

    if sum_d_sq > 0:
        chisq_durkalski = float((sum_d ** 2) / sum_d_sq)
        p_durkalski = float(stats.chi2.sf(chisq_durkalski, df=1))
    else:
        chisq_durkalski = 0.0
        p_durkalski = 1.0

    # Exact McNemar reference (binomial test on discordant pairs)
    n_disc = n10_total + n01_total
    if n_disc > 0:
        k_min = min(n10_total, n01_total)
        # two-sided binomial test with p = 0.5
        p_exact = float(min(1.0, 2.0 * stats.binom.cdf(k_min, n_disc, 0.5)))
    else:
        p_exact = 1.0

    return {
        "n10": n10_total,
        "n01": n01_total,
        "n_clusters": M,
        "chisq_durkalski": round(chisq_durkalski, 4),
        "p_durkalski": p_durkalski,
        "p_exact_mcnemar_ref": p_exact
    }


def holm_correction(p_values):
    """Holm-Bonferroni step-down correction for an array of p-values."""
    m = len(p_values)
    order = np.argsort(p_values)
    adj = np.empty(m)
    running_max = 0.0
    for rank, idx in enumerate(order):
        pval = p_values[idx]
        factor = m - rank
        raw_adj = pval * factor
        running_max = max(running_max, raw_adj)
        adj[idx] = min(1.0, running_max)
    return adj.tolist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--r50", default="e7_r50_oof.csv", help="ResNet50 ImageNet predictions")
    ap.add_argument("--r50-lc", default="e7_r50_lcinit_oof.csv", help="ResNet50 LC25000 init predictions")
    ap.add_argument("--vit", default="e7_vit_oof.csv", help="ViT-S/16 predictions")
    ap.add_argument("--out", default="lunghist_statistical_tests.json")
    args = ap.parse_args()

    # Model pairs as per Section 3.7
    pairs = [
        ("ResNet50 (ImageNet) vs ViT-S/16 (ImageNet)", args.r50, args.vit),
        ("ResNet50 (ImageNet) vs ResNet50 (LC25000 init)", args.r50, args.r50_lc),
        ("ResNet50 (LC25000 init) vs ViT-S/16 (ImageNet)", args.r50_lc, args.vit)
    ]

    def resolve_file(f):
        if not f:
            return f
        if os.path.exists(f):
            return f
        candidates = [
            f,
            os.path.join("results", os.path.basename(f)),
            f.replace("vit", "vit_s16"),
            f.replace("vit_s16", "vit"),
            os.path.join("results", os.path.basename(f).replace("vit", "vit_s16")),
            os.path.join("runs", os.path.basename(f))
        ]
        for c in candidates:
            if os.path.exists(c):
                return c
        return f

    results = []
    for pair_name, raw_f1, raw_f2 in pairs:
        f1, f2 = resolve_file(raw_f1), resolve_file(raw_f2)
        if not os.path.exists(f1) or not os.path.exists(f2):
            print(f"[Skip] {pair_name}: Files not found ({raw_f1} or {raw_f2})")
            continue

        df1 = pd.read_csv(f1)
        df2 = pd.read_csv(f2)

        # Merge on path or index
        if "path" in df1.columns and "path" in df2.columns:
            m = pd.merge(df1, df2, on="path", suffixes=("_1", "_2"))
            y_true = m["label_1"].to_numpy()
            pred1 = m["pred_1"].to_numpy()
            pred2 = m["pred_2"].to_numpy()
            patients = m["patient_1"].to_numpy()
        else:
            y_true = df1["label"].to_numpy()
            pred1 = df1["pred"].to_numpy()
            pred2 = df2["pred"].to_numpy()
            patients = df1["patient"].to_numpy()

        res = clustered_mcnemar_durkalski(y_true, pred1, pred2, patients)
        res["pair"] = pair_name
        res["file1"] = f1
        res["file2"] = f2
        results.append(res)

    if not results:
        print("[Info] No LungHist700 predictions found to compare yet.")
        return

    # Apply Holm correction
    p_raw = [r["p_durkalski"] for r in results]
    p_adj = holm_correction(p_raw)
    for r, padj in zip(results, p_adj):
        r["p_durkalski_holm_adj"] = padj

    with open(args.out, "w", encoding="utf-8") as fp:
        json.dump(results, fp, indent=2)

    txt_out = os.path.splitext(args.out)[0] + ".txt"
    with open(txt_out, "w", encoding="utf-8") as fp:
        fp.write("=" * 80 + "\n")
        fp.write("LUNGHIST700 STATISTICAL TESTS: CLUSTERED MCNEMAR (DURKALSKI ET AL., 2003)\n")
        fp.write("=" * 80 + "\n\n")
        for r in results:
            fp.write(f"Pair: {r['pair']}\n")
            fp.write(f"  Discordant: n10={r['n10']} (M1 correct, M2 incorrect), n01={r['n01']} (M1 incorrect, M2 correct)\n")
            fp.write(f"  Clusters (Patients): {r['n_clusters']}\n")
            fp.write(f"  Durkalski Chi-Square(1): {r['chisq_durkalski']:.4f}\n")
            fp.write(f"  Raw Durkalski p-value  : {r['p_durkalski']:.4e}\n")
            fp.write(f"  Holm-adjusted p-value  : {r['p_durkalski_holm_adj']:.4e}\n")
            fp.write(f"  Exact McNemar p (ref)  : {r['p_exact_mcnemar_ref']:.4e}\n\n")

    print(f"\n[OK] LungHist700 statistical tests saved to {args.out} and {txt_out}:")
    print(open(txt_out).read())


if __name__ == "__main__":
    main()
