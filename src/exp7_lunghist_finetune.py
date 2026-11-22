"""Exp 6 + Exp 7 - LungHist700 fine-tuning under controlled protocol (Section 3.7 & Table 6).

Exp 7 (main): patient-grouped 5-fold CV, inner patient-grouped validation for early stopping,
              per-magnification analysis (20x and 40x). Replaces the old 4-epoch CPU protocol.
Exp 6 (ablation): image-level 5-fold CV (leakage ablation), measuring the image-level vs
                  patient-level gap with identical epochs/augmentation/preprocessing.

Models evaluated:
  1. ResNet-50 (ImageNet-1K V2)
  2. ResNet-50 (LC25000 init from runs/resnet50_main/fold0.pt)
  3. ViT-S/16 (timm augreg, encoder blocks 0 & 1 frozen)

Settings (Section 3.7):
  - Optimizer: Adam 1e-4, wd 1e-4 (ResNet-50); AdamW 1e-4, wd 1e-2 (ViT-S/16)
  - Loss: class-weighted Cross-Entropy, batch size 32
  - Epochs: max 50, patience 8 on inner validation (~15% patients in training fold)
  - Preprocessing: center-crop <= 768 px, resize 224, flips, rotation +-15 deg, mild jitter
  - Metrics: accuracy + Wilson CI, patient-level cluster-bootstrap CI (2000 resamples),
             balanced accuracy, macro F1, macro AUC, recall per class, and acc by magnification.
"""
import argparse
import copy
import json
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as T
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

from PIL import Image
from common import (build_model, load_checkpoint, get_lunghist_root, ImageList,
                    MEAN, STD, metrics, cluster_bootstrap, label_to_int)
from train_scenarios import resolve_split_csv


class HEDJitter:
    """Ruifrok-Johnston HED stain augmentation (Tellez et al., sigma per channel)."""

    # Standard H&E&D optical density matrix and inverse
    HED_FROM_RGB = np.array([
        [1.87798274, -1.00767869, -0.55611582],
        [-0.06590806, 1.13473037, -0.13550307],
        [-0.60190736, -0.48041419, 1.57358807],
    ], dtype=np.float32)
    RGB_FROM_HED = np.array([
        [0.65, 0.70, 0.29],
        [0.07, 0.99, 0.11],
        [0.27, 0.57, 0.78],
    ], dtype=np.float32)

    def __init__(self, sigma=0.05):
        self.sigma = float(sigma)
        # Normalize rows of RGB_FROM_HED
        self.rgb_from_hed = self.RGB_FROM_HED / np.linalg.norm(self.RGB_FROM_HED, axis=1, keepdims=True)
        self.hed_from_rgb = np.linalg.inv(self.rgb_from_hed)

    def __call__(self, img):
        if self.sigma <= 0:
            return img
        rgb = np.asarray(img, dtype=np.float32) / 255.0
        rgb = np.clip(rgb, 1e-6, 1.0)
        od = -np.log(rgb) / np.log(10.0)
        hed = od @ self.hed_from_rgb
        alpha = np.random.uniform(1.0 - self.sigma, 1.0 + self.sigma, size=(1, 1, 3)).astype(np.float32)
        beta = np.random.uniform(-self.sigma, self.sigma, size=(1, 1, 3)).astype(np.float32)
        hed_aug = np.clip(hed * alpha + beta, 0.0, None)
        od_aug = hed_aug @ self.rgb_from_hed
        rgb_aug = np.clip(np.power(10.0, -od_aug), 0.0, 1.0)
        return Image.fromarray((rgb_aug * 255.0).round().astype(np.uint8))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="lunghist700.csv")
    ap.add_argument("--root", default="")
    ap.add_argument("--data_root", default="")
    ap.add_argument("--arch", default="resnet50", choices=["resnet50", "vit_s16", "vit_s"])
    ap.add_argument("--init", default=None, help="checkpoint for LC25000 init (e.g. runs/resnet50_main/fold0.pt)")
    ap.add_argument("--split", default="patient", choices=["patient", "image"])
    ap.add_argument("--folds", default=None, help="reuse existing fold assignment file")
    ap.add_argument("--folds-out", default=None, help="save fold assignment file")
    ap.add_argument("--magnification", default=None, choices=["20x", "40x", None], help="filter by 20x or 40x")
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--hed_jitter", type=float, default=0.0, help="WP06b HED stain jitter sigma (default: 0.0)")
    ap.add_argument("--freeze_vit", type=int, default=1, choices=[0, 1])
    ap.add_argument("--out", required=True)
    ap.add_argument("--no-pretrained", action="store_true")
    args = ap.parse_args()

    arch_name = "vit_s16" if args.arch in ["vit_s16", "vit_s"] else "resnet50"
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    root_dir = get_lunghist_root(args.data_root or args.root)
    csv_path = resolve_split_csv(args.csv)
    folds_path = resolve_split_csv(args.folds) if args.folds else None

    print(f"\n==========================================================================")
    print(f"  LUNGHIST700 FINE-TUNING: {args.out}")
    print(f"  Arch: {arch_name} | Split: {args.split} | Magnification: {args.magnification or 'All (20x + 40x)'}")
    print(f"  Init: {args.init or 'ImageNet-1K'}")
    print(f"  Root: {root_dir}")
    print(f"==========================================================================")

    df = pd.read_csv(csv_path)
    if args.magnification:
        df = df[df["magnification"].astype(str) == args.magnification].reset_index(drop=True)

    y_all = df["label"].map(label_to_int).to_numpy()
    g_all = df["patient"].to_numpy()

    # Manage fold assignments
    if folds_path and os.path.exists(folds_path):
        print(f"Reusing existing fold file: {folds_path}")
        f_df = pd.read_csv(folds_path)
        if "path" in f_df.columns and "path" in df.columns:
            fold_map = dict(zip(f_df["path"], f_df["fold"]))
            df["fold"] = df["path"].map(fold_map).astype(int)
        else:
            df["fold"] = f_df["fold"].to_numpy().astype(int)
    else:
        print(f"Generating 5-fold {args.split} split (seed {args.seed})...")
        if args.split == "patient":
            sgkf = StratifiedGroupKFold(5, shuffle=True, random_state=args.seed)
            sp = sgkf.split(df, y_all, g_all)
        else:
            skf = StratifiedKFold(5, shuffle=True, random_state=args.seed)
            sp = skf.split(df, y_all)

        df["fold"] = -1
        for f, (_, te_idx) in enumerate(sp):
            df.loc[te_idx, "fold"] = f

    if args.folds_out:
        df[["path", "patient", "fold"]].to_csv(args.folds_out, index=False)
        print(f"Fold assignments saved to: {args.folds_out}")

    # Transforms (Section 3.7)
    crop = T.Lambda(lambda im: T.CenterCrop(min(768, *im.size))(im))
    tr_list = [
        crop,
        T.Resize((args.size, args.size)),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomVerticalFlip(p=0.5),
        T.RandomRotation(degrees=15),
        T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.05, hue=0.02),
    ]
    if args.hed_jitter > 0:
        tr_list.append(HEDJitter(args.hed_jitter))
    tr_list.extend([T.ToTensor(), T.Normalize(MEAN, STD)])
    tr_transform = T.Compose(tr_list)
    eval_transform_l = T.Compose([
        crop,
        T.Resize((args.size, args.size)),
        T.ToTensor(),
        T.Normalize(MEAN, STD)
    ])

    def make_model():
        m = build_model(arch_name, num_classes=3, pretrained=not args.no_pretrained, freeze_vit_blocks=bool(args.freeze_vit))
        if args.init:
            load_checkpoint(m, args.init, dev)
        return m

    def run_epoch(model, loader, optimizer=None, criterion=None):
        is_train = optimizer is not None
        model.train(is_train)
        probs_all, targets_all = [], []
        total_loss = 0.0

        with torch.set_grad_enabled(is_train):
            for x, y, _ in loader:
                x, y = x.to(dev), y.to(dev)
                with torch.autocast(dev, dtype=torch.float16, enabled=(dev == "cuda")):
                    out = model(x)
                    loss = criterion(out.float(), y)
                if is_train:
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    optimizer.step()
                total_loss += loss.item() * len(y)
                probs_all.append(out.float().softmax(1).detach().cpu())
                targets_all.append(y.cpu())

        return total_loss / len(loader.dataset), torch.cat(probs_all).numpy(), torch.cat(targets_all).numpy()

    oof_probs = np.zeros((len(df), 3))
    training_log = []

    for f in range(5):
        test_sub = df[df["fold"] == f]
        rest_sub = df[df["fold"] != f].reset_index(drop=True)

        # Inner validation for early stopping (~15% of patients)
        y_rest = rest_sub["label"].map(label_to_int).to_numpy()
        if args.split == "patient":
            n_inner_splits = min(7, rest_sub["patient"].nunique())
            inner_sgkf = StratifiedGroupKFold(n_splits=n_inner_splits, shuffle=True, random_state=args.seed)
            inner_splits = inner_sgkf.split(rest_sub, y_rest, rest_sub["patient"])
        else:
            inner_skf = StratifiedKFold(n_splits=7, shuffle=True, random_state=args.seed)
            inner_splits = inner_skf.split(rest_sub, y_rest)

        tr_idx, val_idx = next(inner_splits)
        tr_df = rest_sub.iloc[tr_idx].reset_index(drop=True)
        val_df = rest_sub.iloc[val_idx].reset_index(drop=True)

        # Leakage verification
        if args.split == "patient":
            tr_pts = set(tr_df["patient"])
            val_pts = set(val_df["patient"])
            te_pts = set(test_sub["patient"])
            assert not (tr_pts & te_pts), f"Patient leakage between train and test in fold {f}!"
            assert not (val_pts & te_pts), f"Patient leakage between val and test in fold {f}!"
            assert not (tr_pts & val_pts), f"Patient leakage between train and val in fold {f}!"

        # Class weights
        counts = np.bincount(tr_df["label"].map(label_to_int), minlength=3)
        counts = np.maximum(counts, 1)
        weights = torch.tensor(len(tr_df) / (3.0 * counts), dtype=torch.float32, device=dev)
        criterion = nn.CrossEntropyLoss(weight=weights)

        model = make_model().to(dev)
        trainable_params = [p for p in model.parameters() if p.requires_grad]

        if arch_name == "resnet50":
            optimizer = torch.optim.Adam(trainable_params, lr=args.lr, weight_decay=1e-4)
        else:
            optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=1e-2)

        mk_loader = lambda d, t, sh: DataLoader(ImageList(d, t, root_dir), batch_size=args.bs, shuffle=sh, num_workers=4)
        train_loader = mk_loader(tr_df, tr_transform, True)
        val_loader = mk_loader(val_df, eval_transform_l, False)
        test_loader = mk_loader(test_sub, eval_transform_l, False)

        best_val_loss = float("inf")
        best_state = copy.deepcopy(model.state_dict())
        best_epoch = 0
        bad_epochs = 0

        for ep in range(1, args.epochs + 1):
            tr_loss, _, _ = run_epoch(model, train_loader, optimizer, criterion)
            val_loss, _, _ = run_epoch(model, val_loader, None, criterion)
            training_log.append({
                "fold": f,
                "epoch": ep,
                "train_loss": round(tr_loss, 4),
                "val_loss": round(val_loss, 4)
            })

            if np.isfinite(val_loss) and val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = copy.deepcopy(model.state_dict())
                best_epoch = ep
                bad_epochs = 0
            else:
                bad_epochs += 1
                if bad_epochs >= args.patience:
                    break

        # Load best checkpoint and evaluate strictly on test fold
        model.load_state_dict(best_state)
        ckpt_save_path = f"{args.out}_fold{f}.pt"
        torch.save(best_state, ckpt_save_path)

        _, te_probs, te_y = run_epoch(model, test_loader, None, criterion)
        oof_probs[test_sub.index] = te_probs
        fold_acc = 100.0 * (te_probs.argmax(axis=1) == te_y).mean()
        print(f"Fold {f}: stopped at epoch {ep} (best: {best_epoch}), test acc {fold_acc:.2f}% (n={len(te_y)})")

    # Aggregate metrics across the entire dataset (out-of-fold)
    pred_all = oof_probs.argmax(axis=1)
    res = metrics(y_all, oof_probs)
    res["acc_cluster_ci"] = cluster_bootstrap(y_all, pred_all, g_all)
    res["recall_per_class"] = [float(100.0 * (pred_all[y_all == k] == k).mean()) for k in range(3)]

    if "magnification" in df.columns:
        res["acc_by_mag"] = {
            str(m): round(float(100.0 * (pred_all[df["magnification"] == m] == y_all[df["magnification"] == m]).mean()), 2)
            for m in df["magnification"].unique()
        }

    res["config"] = vars(args)

    out_df = df.copy()
    out_df[["p_ACA", "p_N", "p_SCC"]] = oof_probs
    out_df["pred"] = pred_all
    out_df.to_csv(args.out + "_oof.csv", index=False)
    pd.DataFrame(training_log).to_csv(args.out + "_log.csv", index=False)

    with open(args.out + "_metrics.json", "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2, default=str)

    print(f"\n[OK] Results saved for {args.out}:")
    print(json.dumps(res, indent=2, default=str))


if __name__ == "__main__":
    main()
