"""WP05 Helper - Export per-image test predictions for the old v2 G1 & G2 models.

Loads existing 5-fold checkpoints for:
  - G1: clean_resnet50/fold{0..4}.pt
  - G2: clean_vit_s16/fold{0..4}.pt
and evaluates them on the group-aware test partition in data_splits/exp4_grouped_split.csv
so bootstrap cluster CIs, effect sizes, and Holm-Bonferroni corrections can be computed
identically alongside WP03 and WP04 runs.
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

SRC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from common import (build_model, load_checkpoint, resolve_checkpoint_path, eval_transform,
                    ImageList, metrics, label_to_int, get_lc25000_root, resolve_split_csv)


def eval_grouped_model(arch, ckpt_dir, test_df, lc_root, out_prefix, dev, bs=32):
    ds = ImageList(test_df, eval_transform("resize"), lc_root)
    dl = DataLoader(ds, batch_size=bs, shuffle=False, num_workers=4, pin_memory=(dev == "cuda"))
    y_test = test_df["label_id"].to_numpy()

    fold_probs = []
    fold_accs = []
    resolved_ckpts = []
    for f in range(5):
        ckpt = resolve_checkpoint_path(os.path.join(ckpt_dir, f"fold{f}.pt"))
        resolved_ckpts.append(ckpt)
        m = load_checkpoint(build_model(arch, num_classes=3, pretrained=False, freeze_vit_blocks=True), ckpt, dev).to(dev).eval()
        p_list = []
        with torch.no_grad(), torch.autocast(dev, dtype=torch.float16, enabled=(dev == "cuda")):
            for x, _, _ in dl:
                out = m(x.to(dev))
                p_list.append(torch.softmax(out.float(), dim=1).cpu().numpy())
        pr = np.concatenate(p_list, axis=0)
        fold_probs.append(pr)
        acc_f = float(100.0 * (pr.argmax(1) == y_test).mean())
        fold_accs.append(round(acc_f, 4))

        f_df = test_df[["path", "label", "label_id"]].copy()
        if "group" in test_df.columns:
            f_df["group"] = test_df["group"].values
        f_df[["p_ACA", "p_N", "p_SCC"]] = pr
        f_df["pred"] = pr.argmax(1)
        f_df.to_csv(f"{out_prefix}_fold{f}_test_preds.csv", index=False)

    ens_prob = np.mean(fold_probs, axis=0)
    ens_res = metrics(y_test, ens_prob)

    ens_df = test_df[["path", "label", "label_id"]].copy()
    if "group" in test_df.columns:
        ens_df["group"] = test_df["group"].values
    ens_df[["p_ACA", "p_N", "p_SCC"]] = ens_prob
    ens_df["pred"] = ens_prob.argmax(1)
    ens_df.to_csv(f"{out_prefix}_ensemble_test_preds.csv", index=False)

    return {
        "arch": arch,
        "ckpt_dir": ckpt_dir,
        "fold_test_acc": fold_accs,
        "ensemble_test": ens_res,
        "n_test": int(len(y_test)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="data_splits/exp4_grouped_split.csv")
    ap.add_argument("--root", default="")
    ap.add_argument("--ckpt-root", default="old_ckpts/runs")
    ap.add_argument("--out", default="results_rev1/WP05")
    ap.add_argument("--bs", type=int, default=32)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    lc_root = get_lc25000_root(args.root)
    csv_path = resolve_split_csv(args.csv)

    df = pd.read_csv(csv_path)
    test_df = df[df["partition"] == "test"].reset_index(drop=True)
    test_df["label_id"] = test_df["label"].map(label_to_int)

    ckpt_base = args.ckpt_root
    if not os.path.isdir(ckpt_base) and os.path.isdir("runs"):
        ckpt_base = "runs"

    summary = {}
    for key, arch, folder in [
        ("G1_clean_resnet50", "resnet50", "clean_resnet50"),
        ("G2_clean_vit_s16", "vit_s16", "clean_vit_s16"),
    ]:
        cdir = os.path.join(ckpt_base, folder)
        probe_ckpt = resolve_checkpoint_path(os.path.join(cdir, "fold0.pt"))
        if not os.path.exists(probe_ckpt):
            raise FileNotFoundError(f"Missing checkpoint for {key} (checked {cdir}/fold0.pt and aliases)")
        resolved_cdir = os.path.dirname(probe_ckpt)
        res = eval_grouped_model(arch, resolved_cdir, test_df, lc_root, os.path.join(args.out, key), dev, args.bs)
        summary[key] = res
        print(f"[OK] {key}: Ensemble Test Acc = {res['ensemble_test']['acc']:.2f}% (n={res['n_test']})")

    with open(os.path.join(args.out, "wp05_g1_g2_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"[OK] WP05 old G1/G2 predictions exported to {args.out}")


if __name__ == "__main__":
    main()
