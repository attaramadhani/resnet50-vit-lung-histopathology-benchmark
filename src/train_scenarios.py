"""Master scenario training pipeline on LC25000 (Section 3.1-3.5, Table 1, 2, 3).
Fully aligned with implementation_plan2.md (revisi 2) approved by Dr. Wahyudi Setiawan:
  - 14 Scenarios (S1-S14, 70 runs total)
  - Dev set: 12,000 images (5-fold stratified CV: 9,600 train / 2,400 val per fold)
  - Test set: 3,000 images (1,000 per class held-out, never accessed during training)
  - Separate per_fold_val.csv (n_val=2400) and per_fold_test.csv (n_test=3000)
  - Early stopping: max 50 epochs, patience 5 on val_loss, min_delta = 0.0
  - Seed: 42 (torch, numpy, random, cudnn.deterministic=True, cudnn.benchmark=False)
  - Provenance: writes config.yaml, history.csv, metrics.json, environment.txt per scenario
"""
import os
import sys
import math
import shutil
import argparse
import time
import json
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import StratifiedKFold
from common import (build_model, train_transform, eval_transform, 
                    ImageList, label_to_int, metrics, count_parameters,
                    get_lc25000_root)

# 14 Scenarios exactly as numbered in implementation_plan2.md (Table Section 3)
SCENARIOS_REGISTRY = {
    "1":  {"name": "resnet50_main",         "model": "ResNet50", "arch": "resnet50", "opt": "Adam",    "lr": 1e-4, "wd": 1e-4, "mom": 0.0},
    "2":  {"name": "resnet50_rmsprop",      "model": "ResNet50", "arch": "resnet50", "opt": "RMSprop", "lr": 1e-4, "wd": 1e-4, "mom": 0.0},
    "3":  {"name": "resnet50_lr_1e3",       "model": "ResNet50", "arch": "resnet50", "opt": "Adam",    "lr": 1e-3, "wd": 1e-4, "mom": 0.0},
    "4":  {"name": "resnet50_lr_1e5",       "model": "ResNet50", "arch": "resnet50", "opt": "Adam",    "lr": 1e-5, "wd": 1e-4, "mom": 0.0},
    "5":  {"name": "resnet50_sgd",          "model": "ResNet50", "arch": "resnet50", "opt": "SGD",     "lr": 1e-4, "wd": 1e-4, "mom": 0.0},
    "6":  {"name": "vit_s_main",            "model": "ViT-S/16", "arch": "vit_s16",  "opt": "AdamW",   "lr": 1e-4, "wd": 1e-4, "mom": 0.0},
    "7":  {"name": "vit_s_rmsprop",         "model": "ViT-S/16", "arch": "vit_s16",  "opt": "RMSprop", "lr": 1e-4, "wd": 1e-4, "mom": 0.0},
    "8":  {"name": "vit_s_lr_1e3",          "model": "ViT-S/16", "arch": "vit_s16",  "opt": "AdamW",   "lr": 1e-3, "wd": 1e-4, "mom": 0.0},
    "9":  {"name": "vit_s_lr_1e5",          "model": "ViT-S/16", "arch": "vit_s16",  "opt": "AdamW",   "lr": 1e-5, "wd": 1e-4, "mom": 0.0},
    "10": {"name": "vit_s_sgd",             "model": "ViT-S/16", "arch": "vit_s16",  "opt": "SGD",     "lr": 1e-4, "wd": 1e-4, "mom": 0.0},
    "11": {"name": "resnet50_sgdm",         "model": "ResNet50", "arch": "resnet50", "opt": "SGD",     "lr": 1e-4, "wd": 1e-4, "mom": 0.9},
    "12": {"name": "vit_s_sgdm",            "model": "ViT-S/16", "arch": "vit_s16",  "opt": "SGD",     "lr": 1e-4, "wd": 1e-4, "mom": 0.9},
    "13": {"name": "resnet50_sgdm_lr_1e3",  "model": "ResNet50", "arch": "resnet50", "opt": "SGD",     "lr": 1e-3, "wd": 1e-4, "mom": 0.9},
    "14": {"name": "vit_s_sgdm_lr_1e3",      "model": "ViT-S/16", "arch": "vit_s16",  "opt": "SGD",     "lr": 1e-3, "wd": 1e-4, "mom": 0.9},
}

DIR_ALIASES = {
    "vit_s_main": ["vit_s16_main"],
    "resnet50_sgdm": ["resnet50_sgdm_1e4"],
    "vit_s_sgdm": ["vit_s16_sgdm_1e4", "vit_s_sgdm_1e4"],
    "resnet50_sgdm_lr_1e3": ["resnet50_sgdm_1e3"],
    "vit_s_sgdm_lr_1e3": ["vit_s16_sgdm_1e3"]
}


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def build_optimizer(model, opt_name, lr, weight_decay, momentum):
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if opt_name == "Adam":
        return torch.optim.Adam(trainable_params, lr=lr, weight_decay=weight_decay)
    elif opt_name == "AdamW":
        return torch.optim.AdamW(trainable_params, lr=lr, weight_decay=weight_decay)
    elif opt_name == "RMSprop":
        return torch.optim.RMSprop(trainable_params, lr=lr, weight_decay=weight_decay, alpha=0.99)
    elif opt_name == "SGD":
        return torch.optim.SGD(trainable_params, lr=lr, weight_decay=weight_decay, momentum=momentum)
    else:
        raise ValueError(f"Unknown optimizer: {opt_name}")


def compute_epoch_lr(base_lr, ep, total_epochs, sched, warmup_epochs):
    if sched == "none":
        return base_lr
    if warmup_epochs > 0 and ep <= warmup_epochs:
        return base_lr * (float(ep) / float(warmup_epochs))
    denom = max(1, total_epochs - warmup_epochs)
    progress = min(1.0, max(0.0, float(ep - warmup_epochs) / float(denom)))
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * progress))


def set_optimizer_lr(optimizer, lr):
    for pg in optimizer.param_groups:
        pg["lr"] = lr


def resolve_split_csv(path):
    if os.path.exists(path):
        return path
    alt = os.path.join("data_splits", os.path.basename(path))
    if os.path.exists(alt):
        return alt
    return path


def sync_aliases(src_dir, alias_names):
    for alias in alias_names:
        dst_dir = os.path.join("runs", alias)
        os.makedirs(dst_dir, exist_ok=True)
        for item in os.listdir(src_dir):
            s = os.path.join(src_dir, item)
            d = os.path.join(dst_dir, item)
            if os.path.isfile(s) and (not os.path.exists(d) or os.path.getmtime(s) > os.path.getmtime(d)):
                shutil.copy2(s, d)


def record_environment(target_dir):
    env_file = os.path.join(target_dir, "environment.txt")
    lines = []
    lines.append(f"Python: {sys.version.split()[0]}")
    lines.append(f"PyTorch: {torch.__version__}")
    if torch.cuda.is_available():
        lines.append(f"Device: {torch.cuda.get_device_name(0)}")
        lines.append(f"CUDA: {torch.version.cuda}")
    try:
        import subprocess
        git_hash = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
        lines.append(f"Git Commit: {git_hash}")
    except Exception:
        pass
    with open(env_file, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default="")
    ap.add_argument("--dev_csv", default="lc25000_dev.csv")
    ap.add_argument("--test_csv", default="lc25000_test.csv")
    ap.add_argument("--scenarios", default="all",
                    help="'all' (1-14), 'main' (1 & 6), 'sgdm' (11-14), or comma-separated scenario numbers/names")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--opt", default=None, choices=[None, "Adam", "AdamW", "RMSprop", "SGD"],
                    help="Override optimizer (default: use scenario definition)")
    ap.add_argument("--lr", type=float, default=None, help="Override peak learning rate")
    ap.add_argument("--wd", type=float, default=None, help="Override weight decay")
    ap.add_argument("--mom", type=float, default=None, help="Override momentum")
    ap.add_argument("--warmup_epochs", type=int, default=0, help="Linear warmup epochs (default: 0)")
    ap.add_argument("--sched", default="none", choices=["none", "cosine"], help="LR schedule (default: none)")
    ap.add_argument("--freeze_vit", type=int, default=1, choices=[0, 1], help="Freeze ViT blocks 0-1 (default: 1)")
    ap.add_argument("--head_dropout", type=float, default=None, help="Override classification head dropout (default: v2 per-arch)")
    ap.add_argument("--out_root", default="runs", help="Root output directory (default: runs)")
    ap.add_argument("--arch", default=None, choices=[None, "resnet50", "vit_s16"],
                    help="Run a single custom architecture instead of SCENARIOS_REGISTRY")
    ap.add_argument("--run_name", default=None, help="Custom output folder name under out_root")
    ap.add_argument("--no_aliases", action="store_true", help="Disable DIR_ALIASES duplication")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    set_seed(args.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    data_root = get_lc25000_root(args.data_root)
    print(f"[Dataset] Resolved LC25000 root directory: '{data_root}'")

    dev_csv_path = resolve_split_csv(args.dev_csv)
    test_csv_path = resolve_split_csv(args.test_csv)

    # 1. Load and Verify Assertions on Splits (Section 1)
    if not os.path.exists(dev_csv_path) or not os.path.exists(test_csv_path):
        raise FileNotFoundError(f"Missing {dev_csv_path} or {test_csv_path}. Check data_splits/.")

    dev_df = pd.read_csv(dev_csv_path)
    test_df = pd.read_csv(test_csv_path)

    dev_df["label_id"] = dev_df["label"].map(label_to_int)
    test_df["label_id"] = test_df["label"].map(label_to_int)

    # Mandatory Assertions from Section 1:
    assert len(dev_df) == 12000, f"Expected exactly 12000 dev images, got {len(dev_df)}"
    assert len(test_df) == 3000, f"Expected exactly 3000 test images, got {len(test_df)}"
    for cls_name in ["ACA", "N", "SCC"]:
        n_dev = (dev_df["label"] == cls_name).sum()
        n_te = (test_df["label"] == cls_name).sum()
        assert n_dev == 4000, f"Expected 4000 {cls_name} in dev, got {n_dev}"
        assert n_te == 1000, f"Expected 1000 {cls_name} in test, got {n_te}"
    assert len(set(dev_df["path"]) & set(test_df["path"])) == 0, "Assertion failed: Data leakage between dev and test!"
    print(f"[Assertions PASSED] Dev: 12,000 (4k/class) | Test: 3,000 (1k/class) | Leakage: 0.0%")

    # 5-fold Stratified CV on Dev (9,600 train / 2,400 val per fold)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.seed)
    dev_folds = list(skf.split(dev_df, dev_df["label_id"]))

    for f_idx, (tr_idx, va_idx) in enumerate(dev_folds):
        assert len(tr_idx) == 9600, f"Expected 9600 train in fold {f_idx}, got {len(tr_idx)}"
        assert len(va_idx) == 2400, f"Expected 2400 val in fold {f_idx}, got {len(va_idx)}"

    # Determine scenarios to run
    if args.arch is not None:
        custom_opt = args.opt if args.opt is not None else ("Adam" if args.arch == "resnet50" else "AdamW")
        custom_lr = args.lr if args.lr is not None else 1e-4
        custom_wd = args.wd if args.wd is not None else 1e-4
        custom_mom = args.mom if args.mom is not None else (0.9 if custom_opt == "SGD" else 0.0)
        custom_name = args.run_name or f"{args.arch}_{custom_opt.lower()}_lr{custom_lr:.0e}_s{args.seed}"
        run_items = [(0, {
            "name": custom_name,
            "model": "ResNet50" if args.arch == "resnet50" else "ViT-S/16",
            "arch": args.arch,
            "opt": custom_opt,
            "lr": custom_lr,
            "wd": custom_wd,
            "mom": custom_mom,
        })]
    else:
        if args.scenarios == "all":
            selected_keys = list(SCENARIOS_REGISTRY.keys())
        elif args.scenarios == "main":
            selected_keys = ["1", "6"]
        elif args.scenarios == "sgdm":
            selected_keys = ["11", "12", "13", "14"]
        else:
            req = [s.strip() for s in args.scenarios.split(",")]
            selected_keys = []
            for r in req:
                for k, v in SCENARIOS_REGISTRY.items():
                    if r == k or r == v["name"]:
                        selected_keys.append(k)
        run_items = []
        for sc_k in selected_keys:
            base_cfg = dict(SCENARIOS_REGISTRY[sc_k])
            if args.opt is not None:
                base_cfg["opt"] = args.opt
            if args.lr is not None:
                base_cfg["lr"] = args.lr
            if args.wd is not None:
                base_cfg["wd"] = args.wd
            if args.mom is not None:
                base_cfg["mom"] = args.mom
            run_items.append((int(sc_k), base_cfg))

    print(f"[Execution] Queued {len(run_items)} scenarios ({len(run_items)*5} runs): {[c['name'] for _, c in run_items]}")
    os.makedirs(args.out_root, exist_ok=True)
    is_legacy_root = (os.path.normpath(args.out_root) == "runs")
    val_csv_out = "per_fold_val.csv" if is_legacy_root else os.path.join(args.out_root, "per_fold_val.csv")
    test_csv_out = "per_fold_test.csv" if is_legacy_root else os.path.join(args.out_root, "per_fold_test.csv")

    val_records = []
    test_records = []
    if os.path.exists(val_csv_out):
        try: val_records = pd.read_csv(val_csv_out).to_dict(orient="records")
        except Exception: pass
    if os.path.exists(test_csv_out):
        try: test_records = pd.read_csv(test_csv_out).to_dict(orient="records")
        except Exception: pass

    def rec_key(r):
        s_val = r.get("seed", 42)
        return f"{r['scenario']}_{r['fold']}" if is_legacy_root else f"{r['scenario']}_s{s_val}_{r['fold']}"

    val_map = {rec_key(r): r for r in val_records}
    test_map = {rec_key(r): r for r in test_records}

    freeze_vit_flag = bool(args.freeze_vit)

    for sc_num, cfg in run_items:
        sc_name = args.run_name if (args.run_name and len(run_items) == 1) else cfg["name"]
        if is_legacy_root:
            out_dir = os.path.join(args.out_root, sc_name)
        elif args.run_name and len(run_items) == 1:
            out_dir = os.path.join(args.out_root, sc_name)
        else:
            out_dir = os.path.join(args.out_root, f"seed_{args.seed}", sc_name)
        os.makedirs(out_dir, exist_ok=True)

        summary_file = os.path.join(out_dir, "metrics.json")
        ens_preds_file = os.path.join(out_dir, "ensemble_test_preds.csv")
        all_ckpts = [os.path.join(out_dir, f"fold{f}.pt") for f in range(5)]

        if os.path.exists(summary_file) and os.path.exists(ens_preds_file) and all(os.path.exists(c) for c in all_ckpts) and not args.force:
            print(f"\n[Skip] Scenario {sc_num}: {sc_name} (seed={args.seed}) already completed. Using cached metrics & preds.")
            if is_legacy_root and not args.no_aliases and sc_name in DIR_ALIASES:
                sync_aliases(out_dir, DIR_ALIASES[sc_name])
            continue

        print(f"\n{'='*75}\nSCENARIO {sc_num}: {sc_name.upper()} (seed={args.seed}) | Model: {cfg['model']} | Opt: {cfg['opt']} | LR: {cfg['lr']} | WD: {cfg['wd']} | Mom: {cfg['mom']} | Sched: {args.sched} (warmup={args.warmup_epochs}) | FreezeViT: {freeze_vit_flag} | HeadDrop: {args.head_dropout}\n{'='*75}")

        # Write config.yaml and environment.txt
        record_environment(out_dir)
        with open(os.path.join(out_dir, "config.yaml"), "w", encoding="utf-8") as f:
            for k_cfg, v_cfg in cfg.items():
                f.write(f"{k_cfg}: {v_cfg}\n")
            f.write(
                f"epochs: {args.epochs}\npatience: {args.patience}\nmin_delta: 0.0\n"
                f"seed: {args.seed}\ngrad_clip: none\nscheduler: {args.sched}\n"
                f"warmup_epochs: {args.warmup_epochs}\nfreeze_vit: {int(freeze_vit_flag)}\n"
                f"head_dropout: {args.head_dropout}\nuse_augmentation: true\n"
            )

        fold_val_accs = []
        fold_test_accs = []
        fold_test_probs = []
        fold_epochs_run = []

        # Train 5 folds on Dev ONLY
        for fold, (train_idx, val_idx) in enumerate(dev_folds):
            set_seed(args.seed)
            fold_dir = os.path.join(out_dir, f"fold_{fold}")
            os.makedirs(fold_dir, exist_ok=True)
            ckpt_path = os.path.join(out_dir, f"fold{fold}.pt")
            hist_csv_path = os.path.join(fold_dir, "history.csv")

            if os.path.exists(ckpt_path) and os.path.exists(hist_csv_path) and not args.force:
                print(f"--- Fold {fold}/4: Checkpoint found. Loading ---")
                m = build_model(cfg["arch"], pretrained=False, freeze_vit_blocks=freeze_vit_flag, head_dropout=args.head_dropout).to(dev)
                m.load_state_dict(torch.load(ckpt_path, map_location=dev))
                df_hist = pd.read_csv(hist_csv_path)
                best_val_loss = float(df_hist["val_loss"].min())
                best_epoch = int(df_hist.loc[df_hist["val_loss"].idxmin(), "epoch"])
                stop_epoch = len(df_hist)
                fold_epochs_run.append(stop_epoch)
            else:
                print(f"\n--- Fold {fold}/4: Training (Train: 9,600, Val: 2,400) ---")
                m = build_model(cfg["arch"], pretrained=True, freeze_vit_blocks=freeze_vit_flag, head_dropout=args.head_dropout).to(dev)
                tot_p, tr_p = count_parameters(m)
                print(f"  Params: Total={tot_p/1e6:.2f}M | Trainable={tr_p/1e6:.2f}M")

                opt = build_optimizer(m, cfg["opt"], cfg["lr"], cfg["wd"], cfg["mom"])
                crit = nn.CrossEntropyLoss()
                scaler = torch.amp.GradScaler(enabled=(dev == "cuda"))

                train_ds = Subset(ImageList(dev_df, train_transform(224), data_root), train_idx)
                train_loader = DataLoader(train_ds, batch_size=args.bs, shuffle=True, num_workers=4, pin_memory=(dev == "cuda"))

                val_ds = Subset(ImageList(dev_df, eval_transform("resize"), data_root), val_idx)
                val_loader = DataLoader(val_ds, batch_size=args.bs, shuffle=False, num_workers=4, pin_memory=(dev == "cuda"))

                best_val_loss = float("inf")
                best_epoch = 0
                stop_epoch = 0
                patience_counter = 0
                history_rows = []

                for ep in range(1, args.epochs + 1):
                    ep_lr = compute_epoch_lr(cfg["lr"], ep, args.epochs, args.sched, args.warmup_epochs)
                    set_optimizer_lr(opt, ep_lr)
                    t_ep_start = time.perf_counter()
                    m.train()
                    tr_loss, tr_corr, tr_tot = 0.0, 0, 0
                    for x, target, _ in train_loader:
                        x, target = x.to(dev), target.to(dev)
                        opt.zero_grad(set_to_none=True)
                        with torch.autocast(dev, dtype=torch.float16, enabled=(dev == "cuda")):
                            out = m(x)
                            loss = crit(out, target)
                        scaler.scale(loss).backward()
                        scaler.step(opt)
                        scaler.update()

                        tr_loss += loss.item() * x.size(0)
                        tr_corr += (out.argmax(1) == target).sum().item()
                        tr_tot += x.size(0)

                    epoch_tr_loss = tr_loss / tr_tot
                    epoch_tr_acc = tr_corr / tr_tot

                    # Validation on 2,400 internal images
                    m.eval()
                    va_loss, va_corr, va_tot = 0.0, 0, 0
                    with torch.no_grad(), torch.autocast(dev, dtype=torch.float16, enabled=(dev == "cuda")):
                        for x, target, _ in val_loader:
                            x, target = x.to(dev), target.to(dev)
                            out = m(x)
                            loss = crit(out, target)
                            va_loss += loss.item() * x.size(0)
                            va_corr += (out.argmax(1) == target).sum().item()
                            va_tot += x.size(0)

                    epoch_va_loss = va_loss / va_tot
                    epoch_va_acc = va_corr / va_tot
                    ep_time = time.perf_counter() - t_ep_start

                    history_rows.append({
                        "epoch": ep,
                        "lr": ep_lr,
                        "train_loss": epoch_tr_loss,
                        "train_acc": epoch_tr_acc,
                        "val_loss": epoch_va_loss,
                        "val_acc": epoch_va_acc,
                        "time_s": round(ep_time, 2)
                    })
                    pd.DataFrame(history_rows).to_csv(hist_csv_path, index=False)

                    print(f"Epoch {ep:02d}/{args.epochs:02d} (lr={ep_lr:.2e}, {ep_time:.1f}s) | Train Acc: {epoch_tr_acc*100:.2f}% | Val Loss: {epoch_va_loss:.4f} Acc: {epoch_va_acc*100:.2f}%")

                    # Section 3.4 Early stopping: min_delta = 0.0 (guard against NaN at extreme LRs)
                    if not os.path.exists(ckpt_path):
                        torch.save(m.state_dict(), ckpt_path)
                    if math.isfinite(epoch_va_loss) and epoch_va_loss < best_val_loss:
                        best_val_loss = epoch_va_loss
                        best_epoch = ep
                        patience_counter = 0
                        torch.save(m.state_dict(), ckpt_path)
                    else:
                        patience_counter += 1
                        if patience_counter >= args.patience:
                            print(f"[Early Stop] Triggered at epoch {ep} (Best: epoch {best_epoch}, val_loss: {best_val_loss:.4f})")
                            break

                stop_epoch = len(history_rows)
                fold_epochs_run.append(stop_epoch)
                m.load_state_dict(torch.load(ckpt_path, map_location=dev))

            # Final validation evaluation on 2,400 fold images
            val_ds_eval = Subset(ImageList(dev_df, eval_transform("resize"), data_root), val_idx)
            val_loader_eval = DataLoader(val_ds_eval, batch_size=args.bs, shuffle=False, num_workers=4, pin_memory=(dev == "cuda"))
            m.eval()
            v_corr = 0
            with torch.no_grad(), torch.autocast(dev, dtype=torch.float16, enabled=(dev == "cuda")):
                for x, target, _ in val_loader_eval:
                    x, target = x.to(dev), target.to(dev)
                    out = m(x)
                    v_corr += (out.argmax(1) == target).sum().item()
            fold_val_acc = (v_corr / 2400.0) * 100.0
            fold_val_accs.append(fold_val_acc)

            # Record per_fold_val.csv and flush immediately to disk
            val_entry = {
                "scenario": sc_name,
                "seed": args.seed,
                "fold": fold,
                "n_val": 2400,
                "correct": v_corr,
                "val_acc": round(fold_val_acc, 4),
                "val_loss": round(best_val_loss, 6),
                "best_epoch": best_epoch,
                "stop_epoch": stop_epoch
            }
            val_map[rec_key(val_entry)] = val_entry
            pd.DataFrame(list(val_map.values())).to_csv(val_csv_out, index=False)
            if is_legacy_root:
                pd.DataFrame(list(val_map.values())).to_csv("per_fold.csv", index=False)

        # 2. Test Set Evaluation: Only after all 5 folds completed (Section 1 & 4)
        print(f"\n[Evaluating on 3,000 Independent Test Images]...")
        test_ds = ImageList(test_df, eval_transform("resize"), data_root)
        test_loader = DataLoader(test_ds, batch_size=args.bs, shuffle=False, num_workers=4, pin_memory=(dev == "cuda"))

        for fold in range(5):
            ckpt_path = os.path.join(out_dir, f"fold{fold}.pt")
            m = build_model(cfg["arch"], pretrained=False, freeze_vit_blocks=freeze_vit_flag, head_dropout=args.head_dropout).to(dev)
            m.load_state_dict(torch.load(ckpt_path, map_location=dev))
            m.eval()
            t_corr = 0
            t_probs_f = []
            with torch.no_grad(), torch.autocast(dev, dtype=torch.float16, enabled=(dev == "cuda")):
                for x, target, _ in test_loader:
                    x = x.to(dev)
                    out = m(x)
                    pr = torch.softmax(out.float(), dim=1).cpu().numpy()
                    t_probs_f.append(pr)
                    t_corr += (pr.argmax(1) == target.numpy()).sum()
            probs_fold = np.concatenate(t_probs_f, axis=0)
            f_test_acc = (t_corr / 3000.0) * 100.0
            fold_test_accs.append(f_test_acc)
            fold_test_probs.append(probs_fold)

            # Save per-image predictions for this fold
            f_pred_df = test_df[["path", "label", "label_id"]].copy()
            f_pred_df[["p_ACA", "p_N", "p_SCC"]] = probs_fold
            f_pred_df["pred"] = probs_fold.argmax(1)
            f_pred_df.to_csv(os.path.join(out_dir, f"fold_{fold}", "test_preds.csv"), index=False)

            test_entry = {
                "scenario": sc_name,
                "seed": args.seed,
                "fold": fold,
                "n_test": 3000,
                "correct": int(t_corr),
                "test_acc": round(f_test_acc, 4)
            }
            test_map[rec_key(test_entry)] = test_entry

        # Soft voting ensemble of 5 folds on 3,000 test images
        ens_probs = np.mean(fold_test_probs, axis=0)
        y_test = test_df["label_id"].to_numpy()
        test_metrics_res = metrics(y_test, ens_probs)

        ens_pred_df = test_df[["path", "label", "label_id"]].copy()
        ens_pred_df[["p_ACA", "p_N", "p_SCC"]] = ens_probs
        ens_pred_df["pred"] = ens_probs.argmax(1)
        ens_pred_df.to_csv(ens_preds_file, index=False)

        # Save metrics.json
        summary = {
            "scenario_number": int(sc_num),
            "scenario": sc_name,
            "seed": args.seed,
            "config": {
                **cfg,
                "epochs": args.epochs,
                "patience": args.patience,
                "sched": args.sched,
                "warmup_epochs": args.warmup_epochs,
                "freeze_vit": int(freeze_vit_flag),
                "head_dropout": args.head_dropout,
            },
            "cv_val_accuracy": {
                "n_val_per_fold": 2400,
                "per_fold": [round(x, 4) for x in fold_val_accs],
                "mean": round(float(np.mean(fold_val_accs)), 2),
                "std": round(float(np.std(fold_val_accs)), 2)
            },
            "fold_test_accuracy": {
                "n_test": 3000,
                "per_fold": [round(x, 4) for x in fold_test_accs],
                "mean": round(float(np.mean(fold_test_accs)), 2),
                "std": round(float(np.std(fold_test_accs)), 2)
            },
            "ensemble_test": test_metrics_res,
            "epochs_completed": fold_epochs_run
        }

        with open(summary_file, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        with open(os.path.join(out_dir, "metrics_summary.json"), "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        if is_legacy_root and not args.no_aliases and sc_name in DIR_ALIASES:
            sync_aliases(out_dir, DIR_ALIASES[sc_name])

        # Write per_fold_val.csv and per_fold_test.csv
        pd.DataFrame(list(val_map.values())).to_csv(val_csv_out, index=False)
        pd.DataFrame(list(test_map.values())).to_csv(test_csv_out, index=False)
        if is_legacy_root:
            pd.DataFrame(list(val_map.values())).to_csv("per_fold.csv", index=False)

        print(f"\n[COMPLETED SCENARIO {sc_num}: {sc_name}]")
        print(f"  CV Val Acc (5x 2,400 val imgs) : {summary['cv_val_accuracy']['mean']:.2f} +/- {summary['cv_val_accuracy']['std']:.2f}%")
        print(f"  Ensemble Test Acc (3,000 imgs) : {test_metrics_res['acc']:.2f}% [95% CI: {test_metrics_res['acc_ci'][0]:.2f}-{test_metrics_res['acc_ci'][1]:.2f}%]")
        print(f"  Macro F1: {test_metrics_res['macro_f1']:.4f} | Macro ROC-AUC: {test_metrics_res['macro_auc']:.6f}")

    print("\n[OK] All scenario training routines completed!")

if __name__ == "__main__":
    main()
