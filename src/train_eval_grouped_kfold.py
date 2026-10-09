"""
Patient-Level 5-Fold Stratified Cross-Validation Pipeline for ResNet-50
Target: Elsevier CMPB Update (Computer Methods and Programs in Biomedicine Update)
Grant: Penelitian Terapan Unggulan Perguruan Tinggi - Hibah BIMA 2026

Features:
1. Patient-Grouped Stratified 5-Fold Cross-Validation using StratifiedGroupKFold on 45 patients (data.csv).
   - Zero patient leakage between training and validation folds.
2. Sequential Transfer Learning: Initialized from ResNet-50 trained on LC25000 (or ImageNet).
3. Evaluates 100% of LungHist700 (all 691 images across 45 patients) via Out-Of-Fold (OOF) inference.
4. Full computational profiling (wall-clock time, latency ms/image, throughput FPS).
5. Comprehensive metrics (Overall OOF Acc, Balanced Acc, Macro F1, Macro ROC-AUC, 5-Fold Mean +- SD).
6. 300 DPI Publication-Grade Visualizations (OOF Confusion Matrix, Multi-Class ROC Curves, Learning Curves).
"""

import os
import sys
import time
import json
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models as tv_models
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report, roc_curve, auc
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

CLASS_NAMES = ["lung_aca", "lung_n", "lung_scc"]
CLASS_LABELS = {
    "lung_aca": "Adenocarcinoma (ACA)",
    "lung_n": "Benign / Normal (N)",
    "lung_scc": "Squamous Cell Ca. (SCC)"
}
NAVY_PRIMARY = "#1B365D"
TEAL_SECONDARY = "#2E6B75"
CORAL_ACCENT = "#D9534F"
PURPLE_ACCENT = "#6F42C1"

class LungHistPatientDataset(Dataset):
    def __init__(self, df: pd.DataFrame, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = row["file_path"]
        image = Image.open(img_path).convert("RGB")
        
        # Center-crop square before resizing to maintain pathology aspect ratio
        w, h = image.size
        crop_sz = min(w, h, 768)
        left = (w - crop_sz) // 2
        top = (h - crop_sz) // 2
        image = image.crop((left, top, left + crop_sz, top + crop_sz))
        
        if self.transform:
            image = self.transform(image)
            
        label = int(row["label_id"])
        return image, label, row["patient_id"], img_path

def build_resnet50_model(num_classes=3, pretrained_weights="imagenet", checkpoint_path=None):
    weights = tv_models.ResNet50_Weights.DEFAULT if pretrained_weights == "imagenet" else None
    model = tv_models.resnet50(weights=weights)
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, num_classes)
    )
    
    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"[INFO] Initializing model weights from checkpoint: {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        state_dict = ckpt.get("model_state_dict", ckpt)
        model_dict = model.state_dict()
        filtered = {k: v for k, v in state_dict.items() if k in model_dict and v.shape == model_dict[k].shape}
        model_dict.update(filtered)
        model.load_state_dict(model_dict)
        print(f"[OK] Successfully transferred {len(filtered)}/{len(model_dict)} parameter tensors.")
    return model

def create_patient_kfold_splits(data_dir: Path, output_csv: Path, n_splits=5, seed=42):
    if output_csv.exists():
        print(f"[INFO] Loading existing patient-level splits from: {output_csv}")
        return pd.read_csv(output_csv)
        
    csv_path = data_dir / "data" / "data.csv"
    if not csv_path.exists():
        csv_path = data_dir / "data.csv"
        
    df_meta = pd.read_csv(csv_path)
    
    def build_fn(row):
        if pd.isna(row["subclass"]) or str(row["subclass"]).strip() == "":
            return f"{row['superclass']}_{row['resolution']}_{row['image_id']}.jpg"
        return f"{row['superclass']}_{row['subclass']}_{row['resolution']}_{row['image_id']}.jpg"
        
    df_meta["filename"] = df_meta.apply(build_fn, axis=1)
    
    # Map physical files
    found_files = {p.name: str(p.resolve()) for p in data_dir.rglob("*.jpg")}
    df_meta["file_path"] = df_meta["filename"].map(found_files)
    
    if df_meta["file_path"].isna().any():
        raise FileNotFoundError(f"Missing images in disk! Matched: {df_meta['file_path'].notna().sum()}/{len(df_meta)}")
        
    class_map = {"aca": 0, "nor": 1, "scc": 2}
    df_meta["label_id"] = df_meta["superclass"].map(class_map)
    df_meta["class_name"] = df_meta["label_id"].map({0: "lung_aca", 1: "lung_n", 2: "lung_scc"})
    
    # StratifiedGroupKFold on patient_id
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    df_meta["fold"] = -1
    for fold_idx, (_, val_idx) in enumerate(sgkf.split(df_meta, df_meta["label_id"], groups=df_meta["patient_id"])):
        df_meta.loc[val_idx, "fold"] = fold_idx + 1
        
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df_meta.to_csv(output_csv, index=False)
    print(f"[OK] Patient-Grouped 5-Fold Splits saved: {output_csv}")
    for f in range(1, n_splits + 1):
        sub = df_meta[df_meta["fold"] == f]
        print(f"  - Fold {f}: {sub['patient_id'].nunique()} patients, {len(sub)} images (ACA: {(sub['label_id']==0).sum()}, N: {(sub['label_id']==1).sum()}, SCC: {(sub['label_id']==2).sum()})")
    return df_meta

def get_transforms():
    train_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    eval_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return train_tf, eval_tf

def plot_confusion_matrix_custom(cm, output_path, title, acc):
    plt.figure(figsize=(7, 6), dpi=300)
    class_labels = ["ACA", "Normal", "SCC"]
    
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    labels_annot = []
    for i in range(3):
        row = []
        for j in range(3):
            row.append(f"{cm[i, j]}\n({cm_norm[i, j]*100:.1f}%)")
        labels_annot.append(row)
        
    sns.heatmap(cm, annot=np.array(labels_annot), fmt='', cmap='Blues', cbar=False,
                xticklabels=class_labels, yticklabels=class_labels, annot_kws={"fontsize": 11, "fontweight": "bold"})
    plt.title(f"{title}\nOut-Of-Fold Accuracy: {acc*100:.2f}% (N=691 Across 45 Patients)", fontsize=11, fontweight='bold', color=NAVY_PRIMARY, pad=12)
    plt.xlabel("Predicted Class", fontsize=11, fontweight='bold')
    plt.ylabel("Ground Truth Class", fontsize=11, fontweight='bold')
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[OK] Confusion Matrix saved: {output_path}")

def plot_multiclass_roc(y_true, y_probs, output_path, macro_auc):
    colors = [CORAL_ACCENT, TEAL_SECONDARY, PURPLE_ACCENT]
    plt.figure(figsize=(8, 6), dpi=300)
    for i, c_name in enumerate(CLASS_NAMES):
        binary_true = (y_true == i).astype(int)
        fpr, tpr, _ = roc_curve(binary_true, y_probs[:, i])
        cls_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, color=colors[i], lw=2.2, label=f"{CLASS_LABELS[c_name]} (AUC = {cls_auc:.4f})")
        
    plt.plot([0, 1], [0, 1], 'k--', lw=1.2, label="Chance Level (AUC = 0.5000)")
    plt.title(f"Patient-Grouped Out-Of-Fold Multi-Class ROC Curves\nResNet-50 (Macro ROC-AUC = {macro_auc:.4f})",
              fontsize=11, fontweight='bold', color=NAVY_PRIMARY, pad=12)
    plt.xlabel("False Positive Rate (1 - Specificity)", fontsize=11, fontweight='bold')
    plt.ylabel("True Positive Rate (Sensitivity)", fontsize=11, fontweight='bold')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(loc="lower right", fontsize=10, frameon=True)
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[OK] Multi-class ROC curves saved: {output_path}")

def run_pipeline(
    data_dir: str,
    output_dir: str,
    base_checkpoint: str = None,
    epochs: int = 4,
    batch_size: int = 32,
    lr: float = 1e-4,
    seed: int = 42
):
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 80)
    print("  PATIENT-LEVEL 5-FOLD STRATIFIED CROSS-VALIDATION PIPELINE (RESNET-50)")
    print("  Sequential Transfer Learning: LC25000 Pretraining -> LungHist700 Fine-Tuning")
    print(f"  Device: {device} | Max Epochs/Fold: {epochs} | Batch Size: {batch_size} | LR: {lr}")
    print("=" * 80)
    
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    splits_csv = out_path / "patient_kfold_splits.csv"
    
    df_splits = create_patient_kfold_splits(Path(data_dir), splits_csv, n_splits=5, seed=seed)
    train_tf, eval_tf = get_transforms()
    
    fold_summaries = []
    oof_predictions = []
    all_fold_histories = []
    
    pipeline_start = time.perf_counter()
    
    for fold in range(1, 6):
        fold_start = time.perf_counter()
        fold_dir = out_path / f"fold_{fold}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        best_ckpt = fold_dir / "best_model.pth"
        
        df_train = df_splits[df_splits["fold"] != fold].copy().reset_index(drop=True)
        df_val = df_splits[df_splits["fold"] == fold].copy().reset_index(drop=True)
        
        train_loader = DataLoader(LungHistPatientDataset(df_train, train_tf), batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(LungHistPatientDataset(df_val, eval_tf), batch_size=batch_size, shuffle=False)
        
        # Calculate inverse class frequency weights
        counts = df_train["label_id"].value_counts().to_dict()
        total_tr = len(df_train)
        weights = [total_tr / (3.0 * counts.get(c, 1)) for c in range(3)]
        class_weights = torch.tensor(weights, dtype=torch.float32).to(device)
        
        # Initialize model (from LC25000 checkpoint if provided, else ImageNet)
        model = build_resnet50_model(num_classes=3, pretrained_weights="imagenet", checkpoint_path=base_checkpoint).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        optimizer = Adam(model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=1)
        
        best_val_acc = -1.0
        fold_history = []
        
        print(f"\n[FOLD {fold}/5] Initiating Training (Train: {len(df_train)} imgs from {df_train['patient_id'].nunique()} patients | Val: {len(df_val)} imgs from {df_val['patient_id'].nunique()} patients)")
        
        for ep in range(1, epochs + 1):
            ep_start = time.perf_counter()
            model.train()
            tr_loss, tr_corr, tr_tot = 0.0, 0, 0
            for imgs, lbls, _, _ in train_loader:
                imgs, lbls = imgs.to(device), lbls.to(device)
                optimizer.zero_grad()
                outs = model(imgs)
                loss = criterion(outs, lbls)
                loss.backward()
                optimizer.step()
                tr_loss += loss.item() * len(lbls)
                preds = outs.argmax(dim=1)
                tr_corr += (preds == lbls).sum().item()
                tr_tot += len(lbls)
                
            tr_loss /= tr_tot
            tr_acc = tr_corr / tr_tot
            
            # Validation
            model.eval()
            val_loss, val_corr, val_tot = 0.0, 0, 0
            with torch.no_grad():
                for imgs, lbls, _, _ in val_loader:
                    imgs, lbls = imgs.to(device), lbls.to(device)
                    outs = model(imgs)
                    loss = criterion(outs, lbls)
                    val_loss += loss.item() * len(lbls)
                    preds = outs.argmax(dim=1)
                    val_corr += (preds == lbls).sum().item()
                    val_tot += len(lbls)
                    
            val_loss /= val_tot
            val_acc = val_corr / val_tot
            dur = time.perf_counter() - ep_start
            fps = (tr_tot + val_tot) / dur if dur > 0 else 0
            scheduler.step(val_acc)
            
            is_best = val_acc > best_val_acc
            mark = " [*BEST]" if is_best else ""
            print(f"  [Fold {fold}] Ep {ep:02d}/{epochs:02d} | Tr Loss: {tr_loss:.4f} | Tr Acc: {tr_acc*100:.2f}% | Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:.2f}% ({dur:.1f}s, {fps:.1f} FPS){mark}")
            
            fold_history.append({
                "fold": fold, "epoch": ep, "train_loss": round(tr_loss, 4), "train_acc": round(tr_acc, 4),
                "val_loss": round(val_loss, 4), "val_acc": round(val_acc, 4), "duration_sec": round(dur, 2)
            })
            
            if is_best:
                best_val_acc = val_acc
                torch.save({"fold": fold, "epoch": ep, "model_state_dict": model.state_dict(), "val_acc": val_acc}, best_ckpt)
                
        all_fold_histories.extend(fold_history)
        pd.DataFrame(fold_history).to_csv(fold_dir / "history.csv", index=False)
        
        # Load best checkpoint for fold evaluation and OOF logging
        ckpt = torch.load(best_ckpt, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        
        f_preds, f_labels, f_probs, f_paths, f_pats = [], [], [], [], []
        with torch.no_grad():
            for imgs, lbls, pats, paths in val_loader:
                imgs = imgs.to(device)
                outs = model(imgs)
                probs = torch.softmax(outs, dim=1).cpu().numpy()
                preds = np.argmax(probs, axis=1)
                f_preds.extend(preds)
                f_labels.extend(lbls.numpy())
                f_probs.extend(probs)
                f_paths.extend(paths)
                f_pats.extend(pats.numpy())
                
        f_acc = accuracy_score(f_labels, f_preds)
        f_bal = balanced_accuracy_score(f_labels, f_preds)
        f_f1 = f1_score(f_labels, f_preds, average='macro', zero_division=0)
        f_auc = roc_auc_score(f_labels, np.array(f_probs), multi_class='ovr', average='macro')
        
        fold_dur = time.perf_counter() - fold_start
        print(f"[OK] Fold {fold} Final Score -> Acc: {f_acc*100:.2f}% | Bal Acc: {f_bal*100:.2f}% | F1: {f_f1:.4f} | AUC: {f_auc:.4f} (Duration: {fold_dur:.1f}s)")
        
        fold_summaries.append({
            "fold": fold, "accuracy_percent": round(f_acc * 100, 2),
            "balanced_accuracy_percent": round(f_bal * 100, 2),
            "macro_f1": round(f_f1, 4), "macro_roc_auc": round(f_auc, 4),
            "total_val_samples": len(df_val), "patients_val": df_val["patient_id"].nunique(),
            "duration_sec": round(fold_dur, 2)
        })
        
        for idx in range(len(f_preds)):
            oof_predictions.append({
                "fold": fold, "patient_id": f_pats[idx], "file_path": f_paths[idx],
                "ground_truth_label": f_labels[idx], "ground_truth_name": CLASS_NAMES[f_labels[idx]],
                "predicted_label": f_preds[idx], "predicted_name": CLASS_NAMES[f_preds[idx]],
                "is_correct": (f_labels[idx] == f_preds[idx]),
                "prob_lung_aca": f_probs[idx][0], "prob_lung_n": f_probs[idx][1], "prob_lung_scc": f_probs[idx][2]
            })
            
    total_pipeline_time = time.perf_counter() - pipeline_start
    df_oof = pd.DataFrame(oof_predictions)
    df_oof.to_csv(out_path / "oof_predictions_691.csv", index=False)
    
    # Global Out-Of-Fold Evaluation over 100% of images (N=691)
    y_true = df_oof["ground_truth_label"].values
    y_pred = df_oof["predicted_label"].values
    y_prob = df_oof[["prob_lung_aca", "prob_lung_n", "prob_lung_scc"]].values
    
    oof_acc = accuracy_score(y_true, y_pred)
    oof_bal = balanced_accuracy_score(y_true, y_pred)
    oof_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    oof_auc = roc_auc_score(y_true, y_prob, multi_class='ovr', average='macro')
    oof_cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    oof_rep = classification_report(y_true, y_pred, target_names=CLASS_NAMES, output_dict=True, zero_division=0)
    
    df_folds = pd.DataFrame(fold_summaries)
    df_folds.to_csv(out_path / "fold_metrics_summary.csv", index=False)
    
    figures_dir = out_path / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    plot_confusion_matrix_custom(oof_cm, str(figures_dir / "patient_kfold_confusion_matrix.png"),
                                "Patient-Grouped Out-Of-Fold Confusion Matrix (ResNet-50)", oof_acc)
    plot_multiclass_roc(y_true, y_prob, str(figures_dir / "patient_kfold_roc_curves.png"), oof_auc)
    
    results = {
        "model": "ResNet-50",
        "pretraining": "LC25000" if base_checkpoint else "ImageNet-1K V2",
        "fine_tuning_cohort": "LungHist700 (100% Out-Of-Fold Evaluation)",
        "total_images": len(df_oof),
        "total_patients": df_splits["patient_id"].nunique(),
        "cross_validation_strategy": "StratifiedGroupKFold (Patient-Level Grouping, 5 Folds)",
        "zero_patient_leakage": True,
        "oof_accuracy_percent": round(oof_acc * 100, 2),
        "oof_balanced_accuracy_percent": round(oof_bal * 100, 2),
        "oof_macro_f1": round(oof_f1, 4),
        "oof_macro_roc_auc": round(oof_auc, 4),
        "fold_mean_accuracy_percent": round(df_folds["accuracy_percent"].mean(), 2),
        "fold_std_accuracy_percent": round(df_folds["accuracy_percent"].std(), 2),
        "fold_mean_macro_f1": round(df_folds["macro_f1"].mean(), 4),
        "fold_std_macro_f1": round(df_folds["macro_f1"].std(), 4),
        "fold_mean_macro_auc": round(df_folds["macro_roc_auc"].mean(), 4),
        "fold_std_macro_auc": round(df_folds["macro_roc_auc"].std(), 4),
        "total_duration_minutes": round(total_pipeline_time / 60, 2),
        "confusion_matrix": oof_cm.tolist(),
        "per_class": {
            cls: {
                "precision": round(oof_rep[cls]["precision"], 4),
                "recall": round(oof_rep[cls]["recall"], 4),
                "f1": round(oof_rep[cls]["f1-score"], 4),
                "support": oof_rep[cls]["support"]
            }
            for cls in CLASS_NAMES
        }
    }
    
    with open(out_path / "patient_kfold_metrics.json", "w") as f:
        json.dump(results, f, indent=2)
        
    print("\n" + "=" * 80)
    print("  PATIENT-LEVEL 5-FOLD CROSS-VALIDATION COMPLETED SUCCESSFULLY!")
    print(f"  TOTAL IMAGES EVALUATED OUT-OF-FOLD: {len(df_oof)} (100% of LungHist700)")
    print(f"  TOTAL PATIENTS: {df_splits['patient_id'].nunique()} (ZERO PATIENT LEAKAGE)")
    print(f"  GLOBAL OUT-OF-FOLD ACCURACY: {oof_acc*100:.2f}% (Previously ~40%!)")
    print(f"  5-FOLD MEAN +- SD ACCURACY : {results['fold_mean_accuracy_percent']}% +- {results['fold_std_accuracy_percent']}%")
    print(f"  GLOBAL MACRO F1-SCORE      : {oof_f1:.4f}")
    print(f"  GLOBAL MACRO ROC-AUC       : {oof_auc:.4f}")
    print(f"  TOTAL PIPELINE DURATION    : {total_pipeline_time/60:.2f} minutes")
    print("=" * 80)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Patient-Level 5-Fold CV on LungHist700")
    parser.add_argument("--data-dir", type=str, default="tahap_7/external_validation/data")
    parser.add_argument("--output-dir", type=str, default="tahap_7/external_generalization_test/runs/patient_kfold_resnet50")
    parser.add_argument("--base-checkpoint", type=str, default="tahap_7/external_generalization_test/runs/best_lc25000_resnet50.pth")
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    args = parser.parse_args()
    
    run_pipeline(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        base_checkpoint=args.base_checkpoint,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr
    )
