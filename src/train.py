import os
import time
import json
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import pandas as pd
from tqdm.auto import tqdm
try:
    from .dataset import LungHistDataset, get_transforms, LABEL2ID, ID2LABEL
    from .models import build_model
except (ImportError, ValueError):
    from src.dataset import LungHistDataset, get_transforms, LABEL2ID, ID2LABEL
    from src.models import build_model

def set_seed(seed: int = 42):
    """Mengunci seluruh generator angka acak untuk reprodusibilitas penuh lintas platform."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_optimizer(model, config):
    opt_name = config.get("optimizer", "AdamW").upper()
    lr = float(config.get("learning_rate", 1e-4))
    wd = float(config.get("weight_decay", 1e-4))

    params = [p for p in model.parameters() if p.requires_grad]

    if opt_name == "ADAMW":
        return torch.optim.AdamW(params, lr=lr, weight_decay=wd)
    elif opt_name == "ADAM":
        return torch.optim.Adam(params, lr=lr, weight_decay=wd)
    elif opt_name == "RMSPROP":
        return torch.optim.RMSprop(params, lr=lr, weight_decay=wd)
    elif opt_name == "SGD":
        momentum = float(config.get("momentum", 0.9))
        return torch.optim.SGD(params, lr=lr, momentum=momentum, weight_decay=wd)
    else:
        raise ValueError(f"Optimizer {opt_name} tidak didukung.")

def train_one_epoch(model, loader, optimizer, criterion, scaler, device, is_vit=True):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()

        dev_type = "cuda" if "cuda" in str(device) else "cpu"
        with torch.amp.autocast(device_type=dev_type, enabled=scaler is not None):
            if is_vit:
                outputs = model(pixel_values=imgs, labels=labels)
                loss = outputs.loss
                logits = outputs.logits
            else:
                logits = model(imgs)
                loss = criterion(logits, labels)

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        running_loss += loss.item() * imgs.size(0)
        preds = torch.argmax(logits, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return running_loss / total, correct / total

def eval_one_epoch(model, loader, criterion, device, is_vit=True):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)

            if is_vit:
                outputs = model(pixel_values=imgs, labels=labels)
                loss = outputs.loss
                logits = outputs.logits
            else:
                logits = model(imgs)
                loss = criterion(logits, labels)

            running_loss += loss.item() * imgs.size(0)
            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return running_loss / total, correct / total

def run_kfold_training(train_df, config, output_dir, device):
    os.makedirs(output_dir, exist_ok=True)
    n_folds = config.get("n_folds", 5)
    epochs = config.get("epochs", 50)
    batch_size = config.get("batch_size", 32)
    patience_es = config.get("patience_es", 5)
    min_delta = float(config.get("min_delta", 1e-4))
    use_fp16 = config.get("fp16", True) and device.type == "cuda"
    num_workers = config.get("num_workers", 2)
    is_vit = "vit" in config.get("model_architecture", "vit_s").lower()

    use_aug = config.get("use_augmentation", False)
    train_transform, eval_transform = get_transforms(
        img_size=config.get("img_size", 224),
        use_augmentation=use_aug
    )
    if use_aug:
        print("[INFO] Data Augmentasi On-The-Fly: DIAKTIFKAN")
    else:
        print("[INFO] Data Augmentasi On-The-Fly: DIMATIKAN (Model dilatih pada citra LC25000 asli tanpa distorsi tambahan)")
    criterion = nn.CrossEntropyLoss()

    seed = int(config.get("seed", 42))
    set_seed(seed)

    n_splits = max(2, n_folds)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    fold_metrics = []

    print(f"\n{'='*60}\nMEMULAI {n_folds}-FOLD CROSS VALIDATION ({config.get('experiment_name')})\n{'='*60}")

    for fold, (train_idx, val_idx) in enumerate(skf.split(train_df["filepath"], train_df["label_id"]), start=1):
        if fold > n_folds:
            break
        print(f"\n>>> FOLD {fold}/{n_folds} <<<")
        fold_dir = os.path.join(output_dir, f"fold_{fold}")
        os.makedirs(fold_dir, exist_ok=True)
        best_ckpt_path = os.path.join(fold_dir, "best_model.pth")
        history_csv_path = os.path.join(fold_dir, "history.csv")

        # Cek fitur Auto-Resume: Lewati jika fold sudah pernah selesai dilatih
        if os.path.exists(best_ckpt_path) and os.path.exists(history_csv_path):
            print(f"[OK] Fold {fold} sudah selesai pada sesi sebelumnya. Melewati ke fold berikutnya...")
            df_hist = pd.read_csv(history_csv_path)
            fold_metrics.append({"fold": fold, "best_val_loss": df_hist["val_loss"].min(), "best_val_acc": df_hist["val_acc"].max()})
            continue

        fold_seed = seed + fold
        set_seed(fold_seed)

        df_train_fold = train_df.iloc[train_idx].reset_index(drop=True)
        df_val_fold = train_df.iloc[val_idx].reset_index(drop=True)

        ds_train = LungHistDataset(df_train_fold, transform=train_transform)
        ds_val = LungHistDataset(df_val_fold, transform=eval_transform)

        pin_memory = (device.type == "cuda")
        g = torch.Generator()
        g.manual_seed(fold_seed)
        loader_train = DataLoader(ds_train, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory, generator=g)
        loader_val = DataLoader(ds_val, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)

        model = build_model(config, LABEL2ID, ID2LABEL).to(device)
        optimizer = get_optimizer(model, config)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=config.get("lr_factor", 0.5), patience=config.get("patience_lr", 2), min_lr=config.get("min_lr", 1e-6)
        )
        try:
            scaler = torch.amp.GradScaler("cuda") if use_fp16 else None
        except Exception:
            scaler = torch.cuda.amp.GradScaler() if use_fp16 else None

        best_val_loss = float("inf")
        best_val_acc = 0.0
        epochs_no_improve = 0
        history_records = []

        start_time = time.perf_counter()

        for ep in range(1, epochs + 1):
            ep_start = time.perf_counter()
            tr_loss, tr_acc = train_one_epoch(model, loader_train, optimizer, criterion, scaler, device, is_vit=is_vit)
            val_loss, val_acc = eval_one_epoch(model, loader_val, criterion, device, is_vit=is_vit)
            scheduler.step(val_loss)
            ep_duration = time.perf_counter() - ep_start
            current_lr = optimizer.param_groups[0]["lr"]

            history_records.append({
                "epoch": ep,
                "train_loss": tr_loss,
                "train_acc": tr_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "lr": current_lr,
                "duration_sec": ep_duration
            })

            print(f"[Fold {fold}][Ep {ep:02d}/{epochs}] Train Loss: {tr_loss:.4f} Acc: {tr_acc:.4f} | Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} | ({ep_duration:.1f}s)")

            # Early stopping & save best model
            if val_loss < (best_val_loss - min_delta):
                best_val_loss = val_loss
                best_val_acc = val_acc
                epochs_no_improve = 0
                torch.save({
                    "fold": fold,
                    "epoch": ep,
                    "model_state_dict": model.state_dict(),
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                    "config": config
                }, best_ckpt_path)
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience_es:
                    print(f"[STOP] Early stopping aktif di Fold {fold} setelah {ep} epoch (tidak membaik selama {patience_es} epoch).")
                    break

        pd.DataFrame(history_records).to_csv(history_csv_path, index=False)
        fold_metrics.append({"fold": fold, "best_val_loss": best_val_loss, "best_val_acc": best_val_acc, "total_time_sec": time.perf_counter() - start_time})
        print(f"[OK] Fold {fold} Selesai: Best Val Acc={best_val_acc*100:.2f}%, Best Val Loss={best_val_loss:.4f}")

    return fold_metrics
