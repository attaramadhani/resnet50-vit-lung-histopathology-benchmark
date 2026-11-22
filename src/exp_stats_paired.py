"""Paired statistics over the 5-fold cross-validation (Section 3.5, 4.1, 4.2).

Computes:
  1. Standard paired t-test over 5 folds (df = 4)
  2. Nadeau-Bengio corrected resampled t-test (r = n_val / n_train = 2400 / 9600 = 0.25, df = 4)
  3. Welch t-test (reported for reference only)

Four mandatory comparisons (Section 4, implementation_plan2.md):
  (a) Scenario 1 vs 6  : resnet50_main vs vit_s_main
  (b) Scenario 5 vs 10 : resnet50_sgd  vs vit_s_sgd
  (c) Scenario 11 vs 12: resnet50_sgdm vs vit_s_sgdm (1e-4)
  (d) Scenario 13 vs 14: resnet50_sgdm_lr_1e3 vs vit_s_sgdm_lr_1e3 (1e-3)

Usage:
  python exp_stats_paired.py                       # runs all 4 standard pairs from per_fold_val.csv
  python exp_stats_paired.py --csv per_fold_val.csv --a resnet50_main --b vit_s_main
"""
import argparse
import json
import math
import os
import numpy as np
import pandas as pd
from scipy import stats

PAIRS = [
    ("Scenario 1 vs 6", "resnet50_main", "vit_s_main", "Adam 1e-4 vs AdamW 1e-4 (Primary)"),
    ("Scenario 5 vs 10", "resnet50_sgd", "vit_s_sgd", "SGD m=0 lr=1e-4"),
    ("Scenario 11 vs 12", "resnet50_sgdm", "vit_s_sgdm", "SGD m=0.9 lr=1e-4 (Exp 5)"),
    ("Scenario 13 vs 14", "resnet50_sgdm_lr_1e3", "vit_s_sgdm_lr_1e3", "SGD m=0.9 lr=1e-3 (Exp 5)")
]

# Aliases for robust scenario lookup across different naming conventions
ALIASES = {
    "vit_s_main": ["vit_s_main", "vit_s16_main", "vit_s16_adamw", "vit_s_adamw"],
    "resnet50_main": ["resnet50_main", "resnet50_adam"],
    "resnet50_sgd": ["resnet50_sgd", "resnet50_sgd_1e4"],
    "vit_s_sgd": ["vit_s_sgd", "vit_s16_sgd", "vit_s_sgd_1e4"],
    "resnet50_sgdm": ["resnet50_sgdm", "resnet50_sgdm_1e4"],
    "vit_s_sgdm": ["vit_s_sgdm", "vit_s16_sgdm", "vit_s_sgdm_1e4"],
    "resnet50_sgdm_lr_1e3": ["resnet50_sgdm_lr_1e3", "resnet50_sgdm_1e3"],
    "vit_s_sgdm_lr_1e3": ["vit_s_sgdm_lr_1e3", "vit_s16_sgdm_lr_1e3", "vit_s_sgdm_1e3"]
}


def find_scenario_data(df, name):
    candidates = ALIASES.get(name, [name])
    for c in candidates:
        sub = df[df["scenario"] == c]
        if len(sub) > 0:
            return c, sub.sort_values("fold")
    return None, None


def compute_comparison(A_vals, B_vals, name_a, name_b, label, n_train=9600, n_test=2400):
    k = len(A_vals)
    d = A_vals - B_vals
    mean_d = float(np.mean(d))
    var_d = float(np.var(d, ddof=1))
    std_d = float(np.std(d, ddof=1))

    # Standard paired t-test
    if var_d > 0:
        se_paired = math.sqrt(var_d / k)
        t_paired = mean_d / se_paired
        p_paired = float(2 * stats.t.sf(abs(t_paired), df=k - 1))
    else:
        t_paired, p_paired = float("nan"), 1.0

    # Nadeau-Bengio corrected resampled t-test
    r = n_test / n_train
    if var_d > 0:
        se_corrected = math.sqrt((1.0 / k + r) * var_d)
        t_corr = mean_d / se_corrected
        p_corr = float(2 * stats.t.sf(abs(t_corr), df=k - 1))
    else:
        t_corr, p_corr = float("nan"), 1.0

    # Welch t-test (reference only)
    tw, pw = stats.ttest_ind(A_vals, B_vals, equal_var=False)

    return {
        "label": label,
        "scenario_a": name_a,
        "scenario_b": name_b,
        "mean_a": round(float(np.mean(A_vals)), 4),
        "std_a": round(float(np.std(A_vals, ddof=1)), 4),
        "mean_b": round(float(np.mean(B_vals)), 4),
        "std_b": round(float(np.std(B_vals, ddof=1)), 4),
        "mean_diff": round(mean_d, 4),
        "paired_t": round(float(t_paired), 3),
        "paired_p": round(float(p_paired), 5),
        "corrected_resampled_t": round(float(t_corr), 3),
        "corrected_resampled_p": round(float(p_corr), 5),
        "welch_t_ref": round(float(tw), 3),
        "welch_p_ref": round(float(pw), 5),
        "n_train": n_train,
        "n_test": n_test,
        "ratio_r": r
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=None, help="per_fold_val.csv or per_fold.csv")
    ap.add_argument("--a", default=None, help="scenario A name")
    ap.add_argument("--b", default=None, help="scenario B name")
    ap.add_argument("--n-train", type=int, default=9600)
    ap.add_argument("--n-test", type=int, default=2400)
    ap.add_argument("--out", default="paired_stats.csv")
    args = ap.parse_args()

    csv_file = args.csv
    if not csv_file:
        for c in ["per_fold_val.csv", "per_fold.csv"]:
            if os.path.exists(c):
                csv_file = c
                break
    if not csv_file or not os.path.exists(csv_file):
        raise FileNotFoundError(f"Cannot find per-fold CSV file. Checked per_fold_val.csv and per_fold.csv.")

    df = pd.read_csv(csv_file)
    # Determine accuracy column
    acc_col = "val_acc"
    if acc_col not in df.columns:
        if "val_acc_2400" in df.columns:
            acc_col = "val_acc_2400"
        elif "acc" in df.columns:
            acc_col = "acc"
        else:
            raise KeyError(f"No validation accuracy column found in {csv_file}. Columns: {list(df.columns)}")

    results = []

    if args.a and args.b:
        # User specified custom pair
        real_a, sub_a = find_scenario_data(df, args.a)
        real_b, sub_b = find_scenario_data(df, args.b)
        if not real_a or not real_b:
            raise ValueError(f"Could not find data for {args.a} (found: {real_a}) or {args.b} (found: {real_b}) in {csv_file}")
        res = compute_comparison(sub_a[acc_col].to_numpy(), sub_b[acc_col].to_numpy(), real_a, real_b, f"{args.a} vs {args.b}", args.n_train, args.n_test)
        results.append(res)
    else:
        # Run all 4 mandatory comparisons
        print(f"\n==========================================================================")
        print(f"  STATISTICAL TESTS OVER 5-FOLD CV (Nadeau-Bengio r = {args.n_test}/{args.n_train} = 0.25)")
        print(f"  Source file: {csv_file}")
        print(f"==========================================================================")
        for label, sc_a, sc_b, desc in PAIRS:
            real_a, sub_a = find_scenario_data(df, sc_a)
            real_b, sub_b = find_scenario_data(df, sc_b)
            if real_a and real_b and len(sub_a) == len(sub_b) == 5:
                res = compute_comparison(sub_a[acc_col].to_numpy(), sub_b[acc_col].to_numpy(), real_a, real_b, f"{label} ({desc})", args.n_train, args.n_test)
                results.append(res)
            else:
                print(f"[Skip] Comparison {label}: data not fully available ({sc_a}: {len(sub_a) if sub_a is not None else 0} folds, {sc_b}: {len(sub_b) if sub_b is not None else 0} folds)")

    if not results:
        print("[Warn] No comparisons could be computed.")
        return

    out_df = pd.DataFrame(results)
    out_df.to_csv(args.out, index=False)
    json_path = os.path.splitext(args.out)[0] + ".json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    for r in results:
        print(f"\n>>> {r['label']}")
        print(f"    {r['scenario_a']}: {r['mean_a']:.2f} +/- {r['std_a']:.2f}%")
        print(f"    {r['scenario_b']}: {r['mean_b']:.2f} +/- {r['std_b']:.2f}%")
        print(f"    Mean Difference : {r['mean_diff']:+.2f}%")
        print(f"    Paired t(4)     = {r['paired_t']:.3f}, p = {r['paired_p']:.4g}")
        print(f"    Corrected t(4)  = {r['corrected_resampled_t']:.3f}, p = {r['corrected_resampled_p']:.4g} (Nadeau-Bengio)")
        print(f"    Welch t (ref)   = {r['welch_t_ref']:.3f}, p = {r['welch_p_ref']:.4g}")

    print(f"\n[OK] Results written to {args.out} and {json_path}")


if __name__ == "__main__":
    main()
