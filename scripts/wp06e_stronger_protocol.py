"""WP06(e) - Stronger LungHist700 Protocol & Authors' 80/10/10 Protocol Reproduction.

Implements Reviewer 5.6 (WP06e):
  1. Native-resolution tiling: 512x512 non-overlapping centered grid tiles from each
     1200x1600 image (2x3 = 6 tiles per image), resized to 224x224, evaluated by
     averaging tile softmax probabilities per image on the 5 patient-grouped folds
     (data_splits/folds_patient.csv).
  2. Inner-fold LR sweep {3e-5, 1e-4, 3e-4} with cosine schedule, chosen strictly on
     the inner patient-grouped validation loss (zero test leakage).
  3. Original LungHist700 authors' protocol reproduction: patient-grouped 80/10/10
     split per magnification (20x & 40x), resize 300x400, ResNet50, reported across
     split 0 and 5-split rotation average for comparison with published 90%/82%.

Saves all checkpoints (.pt), per-image OOF/test predictions, training logs, and JSON metrics.
"""
import argparse
import copy
import json
import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import Dataset, DataLoader

SRC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from common import (build_model, MEAN, STD, metrics, cluster_bootstrap,
                    label_to_int, get_lunghist_root)
from train_scenarios import compute_epoch_lr, set_optimizer_lr, resolve_split_csv


def extract_nonoverlap_512_tiles(im):
    w, h = im.size
    cols = max(1, w // 512)
    rows = max(1, h // 512)
    if w < 512 or h < 512:
        return [im]
    crop_w, crop_h = cols * 512, rows * 512
    left0 = (w - crop_w) // 2
    top0 = (h - crop_h) // 2
    tiles = []
    for r in range(rows):
        for c in range(cols):
            box = (left0 + c * 512, top0 + r * 512, left0 + (c + 1) * 512, top0 + (r + 1) * 512)
            tiles.append(im.crop(box))
    return tiles


class TileDataset(Dataset):
    def __init__(self, df, root, transform):
        self.samples = []
        self.t = transform
        self.root = root
        for img_idx, r in df.reset_index(drop=True).iterrows():
            rel = str(r["path"]).replace("\\", "/").lstrip("/")
            full_p = os.path.join(root, rel)
            with Image.open(full_p) as im:
                w, h = im.size
            n_tiles = max(1, w // 512) * max(1, h // 512) if (w >= 512 and h >= 512) else 1
            y = label_to_int(r["label"])
            for t_idx in range(n_tiles):
                self.samples.append((full_p, t_idx, y, img_idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        full_p, t_idx, y, img_idx = self.samples[i]
        with Image.open(full_p) as im:
            im = im.convert("RGB")
            tiles = extract_nonoverlap_512_tiles(im)
            tile = tiles[min(t_idx, len(tiles) - 1)]
        return self.t(tile), y, img_idx


class ResizeWHDataset(Dataset):
    def __init__(self, df, root, transform):
        self.df = df.reset_index(drop=True)
        self.root = root
        self.t = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        rel = str(r["path"]).replace("\\", "/").lstrip("/")
        im = Image.open(os.path.join(self.root, rel)).convert("RGB")
        return self.t(im), label_to_int(r["label"]), i


def train_eval_one(model, tr_loader, va_loader, crit, opt_name, lr, wd, epochs, patience, dev):
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=lr, weight_decay=wd) if opt_name == "Adam" else torch.optim.AdamW(params, lr=lr, weight_decay=wd)
    scaler = torch.amp.GradScaler(enabled=(dev == "cuda"))
    best_loss, best_state, bad, best_ep = float("inf"), copy.deepcopy(model.state_dict()), 0, 0

    for ep in range(1, epochs + 1):
        ep_lr = compute_epoch_lr(lr, ep, epochs, "cosine", warmup_epochs=min(3, max(1, epochs // 10)))
        set_optimizer_lr(opt, ep_lr)
        model.train()
        for x, y, _ in tr_loader:
            x, y = x.to(dev), y.to(dev)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(dev, dtype=torch.float16, enabled=(dev == "cuda")):
                loss = crit(model(x).float(), y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

        model.eval()
        va_loss, n_va = 0.0, 0
        with torch.no_grad(), torch.autocast(dev, dtype=torch.float16, enabled=(dev == "cuda")):
            for x, y, _ in va_loader:
                x, y = x.to(dev), y.to(dev)
                loss = crit(model(x).float(), y)
                va_loss += loss.item() * len(y)
                n_va += len(y)
        va_loss /= max(1, n_va)
        if np.isfinite(va_loss) and va_loss < best_loss:
            best_loss = va_loss
            best_state = copy.deepcopy(model.state_dict())
            best_ep = ep
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break
    return best_state, best_loss, best_ep


def predict_tile_image_probs(model, tile_loader, n_images, dev):
    model.eval()
    sums = np.zeros((n_images, 3), dtype=np.float64)
    counts = np.zeros(n_images, dtype=np.int64)
    with torch.no_grad(), torch.autocast(dev, dtype=torch.float16, enabled=(dev == "cuda")):
        for x, _, img_idx in tile_loader:
            pr = torch.softmax(model(x.to(dev)).float(), dim=1).cpu().numpy()
            for row_p, idx_i in zip(pr, img_idx.numpy()):
                sums[idx_i] += row_p
                counts[idx_i] += 1
    return sums / np.maximum(counts[:, None], 1)


def run_tiled_inner_lr(df, lh_root, arch, out_dir, dev, bs=32, epochs=35, patience=6, seed=42):
    os.makedirs(out_dir, exist_ok=True)
    tr_tf = T.Compose([
        T.Resize((224, 224)),
        T.RandomHorizontalFlip(0.5),
        T.RandomVerticalFlip(0.5),
        T.RandomRotation(15),
        T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.05, hue=0.02),
        T.ToTensor(),
        T.Normalize(MEAN, STD),
    ])
    ev_tf = T.Compose([T.Resize((224, 224)), T.ToTensor(), T.Normalize(MEAN, STD)])

    lrs = [3e-5, 1e-4, 3e-4]
    opt_name = "Adam" if arch == "resnet50" else "AdamW"
    wd = 1e-4 if arch == "resnet50" else 1e-2

    oof_probs = np.zeros((len(df), 3), dtype=np.float64)
    fold_records = []

    for f in range(5):
        test_sub = df[df["fold"] == f].reset_index(drop=True)
        test_orig_idx = df.index[df["fold"] == f].to_numpy()
        ckpt_p = os.path.join(out_dir, f"wp06e_tile_{arch}_fold{f}.pt")
        fold_prob_p = os.path.join(out_dir, f"wp06e_tile_{arch}_fold{f}_probs.npy")
        fold_meta_p = os.path.join(out_dir, f"wp06e_tile_{arch}_fold{f}_meta.json")

        if os.path.exists(ckpt_p) and os.path.exists(fold_prob_p) and os.path.exists(fold_meta_p):
            oof_probs[test_orig_idx] = np.load(fold_prob_p)
            with open(fold_meta_p, "r", encoding="utf-8") as fp:
                rec = json.load(fp)
            fold_records.append(rec)
            print(f"[Skip - Resume] WP06e Tile {arch} Fold {f} already completed (test_acc={rec['test_acc']}%)")
            continue

        rest_sub = df[df["fold"] != f].reset_index(drop=True)
        y_rest = rest_sub["label"].map(label_to_int).to_numpy()

        n_inner = min(7, rest_sub["patient"].nunique())
        inner_sgkf = StratifiedGroupKFold(n_splits=n_inner, shuffle=True, random_state=seed)
        tr_idx, va_idx = next(inner_sgkf.split(rest_sub, y_rest, rest_sub["patient"]))
        tr_df = rest_sub.iloc[tr_idx].reset_index(drop=True)
        va_df = rest_sub.iloc[va_idx].reset_index(drop=True)

        assert not (set(tr_df["patient"]) & set(test_sub["patient"]))
        assert not (set(va_df["patient"]) & set(test_sub["patient"]))
        assert not (set(tr_df["patient"]) & set(va_df["patient"]))

        counts = np.maximum(np.bincount(tr_df["label"].map(label_to_int), minlength=3), 1)
        weights = torch.tensor(len(tr_df) / (3.0 * counts), dtype=torch.float32, device=dev)
        crit = nn.CrossEntropyLoss(weight=weights)

        tr_loader = DataLoader(TileDataset(tr_df, lh_root, tr_tf), batch_size=bs, shuffle=True, num_workers=4, pin_memory=(dev == "cuda"))
        va_loader = DataLoader(TileDataset(va_df, lh_root, ev_tf), batch_size=bs, shuffle=False, num_workers=4, pin_memory=(dev == "cuda"))
        te_loader = DataLoader(TileDataset(test_sub, lh_root, ev_tf), batch_size=bs, shuffle=False, num_workers=4, pin_memory=(dev == "cuda"))

        best_cand_loss = float("inf")
        best_lr, best_state, best_ep = lrs[0], None, 0
        lr_losses = {}

        for cand_lr in lrs:
            torch.manual_seed(seed)
            m = build_model(arch, num_classes=3, pretrained=True, freeze_vit_blocks=True).to(dev)
            st, v_loss, b_ep = train_eval_one(m, tr_loader, va_loader, crit, opt_name, cand_lr, wd, epochs, patience, dev)
            lr_losses[str(cand_lr)] = round(float(v_loss), 4)
            if best_state is None or (np.isfinite(v_loss) and v_loss < best_cand_loss):
                best_cand_loss = v_loss
                best_lr = cand_lr
                best_state = st
                best_ep = b_ep

        torch.save(best_state, ckpt_p)
        eval_m = build_model(arch, num_classes=3, pretrained=False, freeze_vit_blocks=True).to(dev)
        eval_m.load_state_dict(best_state)
        probs = predict_tile_image_probs(eval_m, te_loader, len(test_sub), dev)
        oof_probs[test_orig_idx] = probs
        np.save(fold_prob_p, probs)
        f_acc = float(100.0 * (probs.argmax(1) == test_sub["label"].map(label_to_int).to_numpy()).mean())
        rec = {"fold": f, "selected_lr": best_lr, "inner_val_loss": lr_losses, "best_epoch": best_ep, "test_acc": round(f_acc, 2)}
        with open(fold_meta_p, "w", encoding="utf-8") as fp:
            json.dump(rec, fp, indent=2)
        fold_records.append(rec)
        print(f"[WP06e Tile {arch}] Fold {f}: selected_lr={best_lr}, inner_val_losses={lr_losses}, test_acc={f_acc:.2f}%")

    y_all = df["label"].map(label_to_int).to_numpy()
    pred_all = oof_probs.argmax(1)
    res = metrics(y_all, oof_probs)
    res["acc_cluster_ci"] = cluster_bootstrap(y_all, pred_all, df["patient"].to_numpy())
    res["recall_per_class"] = [float(100.0 * (pred_all[y_all == k] == k).mean()) for k in range(3)]
    res["acc_by_mag"] = {
        str(m): round(float(100.0 * (pred_all[df["magnification"] == m] == y_all[df["magnification"] == m]).mean()), 2)
        for m in df["magnification"].unique()
    }
    res["fold_selection"] = fold_records

    out_df = df.copy()
    out_df[["p_ACA", "p_N", "p_SCC"]] = oof_probs
    out_df["pred"] = pred_all
    out_df.to_csv(os.path.join(out_dir, f"wp06e_tile_{arch}_oof.csv"), index=False)
    with open(os.path.join(out_dir, f"wp06e_tile_{arch}_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2)
    return res


def run_orig_80_10_10(df, lh_root, out_dir, dev, bs=32, epochs=50, patience=8, seed=42):
    os.makedirs(out_dir, exist_ok=True)
    # 300x400 (H=300, W=400) matching original LungHist700 aspect ratio (1200x1600 -> 1/4)
    tr_tf = T.Compose([
        T.Resize((300, 400)),
        T.RandomHorizontalFlip(0.5),
        T.RandomVerticalFlip(0.5),
        T.ToTensor(),
        T.Normalize(MEAN, STD),
    ])
    ev_tf = T.Compose([T.Resize((300, 400)), T.ToTensor(), T.Normalize(MEAN, STD)])

    from sklearn.model_selection import GroupShuffleSplit

    orig_summary = {}
    for mag in ["20x", "40x"]:
        sub = df[df["magnification"].astype(str) == mag].reset_index(drop=True)
        y_sub = sub["label"].map(label_to_int).to_numpy()
        g_sub = sub["patient"].to_numpy()

        # Find 5 deterministic 80/10/10 patient-grouped splits where train, val, and test all contain all 3 classes
        collected = []
        cand_seed = seed
        while len(collected) < 5 and cand_seed < seed + 2000:
            gss1 = GroupShuffleSplit(n_splits=1, test_size=0.12, random_state=cand_seed)
            rest_idx, te_idx = next(gss1.split(sub, y_sub, g_sub))
            cand_seed += 1
            if set(np.unique(y_sub[te_idx])) != {0, 1, 2}:
                continue
            gss2 = GroupShuffleSplit(n_splits=1, test_size=0.12, random_state=cand_seed * 7)
            r_tr, r_va = next(gss2.split(sub.iloc[rest_idx], y_sub[rest_idx], g_sub[rest_idx]))
            tr_idx = rest_idx[r_tr]
            va_idx = rest_idx[r_va]
            if set(np.unique(y_sub[va_idx])) != {0, 1, 2} or set(np.unique(y_sub[tr_idx])) != {0, 1, 2}:
                continue
            collected.append((tr_idx, va_idx, te_idx))

        assert len(collected) == 5, f"Could not find 5 valid 3-class 80/10/10 splits for {mag}"
        split_results = []

        for rep, (tr_idx, va_idx, te_idx) in enumerate(collected):

            tr_df = sub.iloc[tr_idx].reset_index(drop=True)
            va_df = sub.iloc[va_idx].reset_index(drop=True)
            te_df = sub.iloc[te_idx].reset_index(drop=True)

            assert not (set(tr_df["patient"]) & set(te_df["patient"]))
            assert not (set(tr_df["patient"]) & set(va_df["patient"]))
            assert not (set(va_df["patient"]) & set(te_df["patient"]))

            counts = np.maximum(np.bincount(tr_df["label"].map(label_to_int), minlength=3), 1)
            weights = torch.tensor(len(tr_df) / (3.0 * counts), dtype=torch.float32, device=dev)
            crit = nn.CrossEntropyLoss(weight=weights)

            torch.manual_seed(seed + rep)
            m = build_model("resnet50", num_classes=3, pretrained=True).to(dev)
            tr_loader = DataLoader(ResizeWHDataset(tr_df, lh_root, tr_tf), batch_size=bs, shuffle=True, num_workers=4, pin_memory=(dev == "cuda"))
            va_loader = DataLoader(ResizeWHDataset(va_df, lh_root, ev_tf), batch_size=bs, shuffle=False, num_workers=4, pin_memory=(dev == "cuda"))
            te_loader = DataLoader(ResizeWHDataset(te_df, lh_root, ev_tf), batch_size=bs, shuffle=False, num_workers=4, pin_memory=(dev == "cuda"))

            best_state, _, best_ep = train_eval_one(m, tr_loader, va_loader, crit, "Adam", 1e-4, 1e-4, epochs, patience, dev)
            ckpt_p = os.path.join(out_dir, f"wp06e_orig801010_{mag}_split{rep}_resnet50.pt")
            torch.save(best_state, ckpt_p)

            m.load_state_dict(best_state)
            m.eval()
            p_list = []
            with torch.no_grad(), torch.autocast(dev, dtype=torch.float16, enabled=(dev == "cuda")):
                for x, _, _ in te_loader:
                    p_list.append(torch.softmax(m(x.to(dev)).float(), 1).cpu().numpy())
            te_prob = np.concatenate(p_list, axis=0)
            y_te = te_df["label"].map(label_to_int).to_numpy()
            r_met = metrics(y_te, te_prob)

            pred_df = te_df.copy()
            pred_df[["p_ACA", "p_N", "p_SCC"]] = te_prob
            pred_df["pred"] = te_prob.argmax(1)
            pred_df.to_csv(os.path.join(out_dir, f"wp06e_orig801010_{mag}_split{rep}_preds.csv"), index=False)

            split_results.append({
                "split": rep,
                "n_train": int(len(tr_df)),
                "n_val": int(len(va_df)),
                "n_test": int(len(te_df)),
                "best_epoch": int(best_ep),
                "acc": round(r_met["acc"], 2),
                "macro_f1": round(r_met["macro_f1"], 4),
                "macro_auc": round(r_met["macro_auc"], 4),
            })

        accs = [s["acc"] for s in split_results]
        orig_summary[mag] = {
            "split0_acc": accs[0],
            "mean_5splits_acc": round(float(np.mean(accs)), 2),
            "std_5splits_acc": round(float(np.std(accs, ddof=1)), 2),
            "splits": split_results,
        }
        print(f"[WP06e Orig 80/10/10] Mag={mag} (300x400 ResNet50): split0={accs[0]:.2f}% | 5-split mean={orig_summary[mag]['mean_5splits_acc']:.2f} +/- {orig_summary[mag]['std_5splits_acc']:.2f}%")

    with open(os.path.join(out_dir, "wp06e_orig_protocol_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(orig_summary, f, indent=2)
    return orig_summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="data_splits/lunghist700.csv")
    ap.add_argument("--folds", default="data_splits/folds_patient.csv")
    ap.add_argument("--root", default="")
    ap.add_argument("--out", default="results_rev1/WP06/wp06e")
    ap.add_argument("--mode", default="all", choices=["all", "tile", "orig"])
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    lh_root = get_lunghist_root(args.root)
    df = pd.read_csv(resolve_split_csv(args.csv))
    f_df = pd.read_csv(resolve_split_csv(args.folds))
    df["fold"] = df["path"].map(dict(zip(f_df["path"], f_df["fold"]))).astype(int)

    if args.mode in ("all", "tile"):
        for arch in ["resnet50", "vit_s16"]:
            run_tiled_inner_lr(df, lh_root, arch, args.out, dev, bs=args.bs, seed=args.seed)
    if args.mode in ("all", "orig"):
        run_orig_80_10_10(df, lh_root, args.out, dev, bs=args.bs, seed=args.seed)


if __name__ == "__main__":
    main()
