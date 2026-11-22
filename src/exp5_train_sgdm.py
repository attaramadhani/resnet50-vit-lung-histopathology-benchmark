"""Exp 5 - Train SGD vs SGD + Momentum (momentum in [0.0, 0.9], lr in [1e-4, 1e-3]) on LC25000 (5-fold CV).
Delegates to train_scenarios.py with --scenarios sgdm or custom list.
"""
import sys
import os
import json
import pandas as pd
import subprocess

if __name__ == "__main__":
    # If called directly, delegate to train_scenarios with sgdm
    sc_list = "resnet50_sgd,resnet50_sgdm,vit_s_sgd,vit_s_sgdm"
    cmd = [sys.executable, "train_scenarios.py", "--scenarios", sc_list] + sys.argv[1:]
    print(f"[Exp 5 Delegating] {' '.join(cmd)}")
    ret = subprocess.call(cmd)
    
    # Also compile exp5_sgdm_metrics.json and per_fold_sgdm.csv for backward compatibility
    if os.path.exists("per_fold.csv"):
        df = pd.read_csv("per_fold.csv")
        df_sgd = df[df["scenario"].isin(["resnet50_sgd", "resnet50_sgdm", "vit_s_sgd", "vit_s_sgdm"])].copy()
        df_sgd.to_csv("per_fold_sgdm.csv", index=False)
        
        metrics_dict = {}
        for sc, grp in df_sgd.groupby("scenario"):
            val_col = "val_acc_2400" if "val_acc_2400" in grp.columns else "val_acc"
            metrics_dict[sc] = {
                "mean_acc": float(grp[val_col].mean()),
                "std_acc": float(grp[val_col].std()),
                "per_fold": [float(x) for x in grp[val_col].tolist()]
            }
        with open("exp5_sgdm_metrics.json", "w", encoding="utf-8") as f:
            json.dump(metrics_dict, f, indent=2)
        print(f"[OK] exp5_sgdm_metrics.json and per_fold_sgdm.csv generated from per_fold.csv")

    sys.exit(ret)
