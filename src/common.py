"""Shared helpers for the revision experiments (CMPB Update manuscript, ResNet50 vs ViT-S/16).

Fully aligned with implementation_plan2.md and Section 3.3 of the manuscript:
  - ResNet-50: ImageNet-1K V2 weights, Dropout(0.3) head -> 23.51 M params
  - ViT-S/16: timm augreg in21k -> in1k, Encoder Blocks 0 & 1 frozen -> 21.67 M params
  - Normalization: ImageNet statistics (MEAN=(0.485, 0.456, 0.406), STD=(0.229, 0.224, 0.225))
  - Dynamic dataset path resolution with 256px cache support
"""
import math
import os
import numpy as np
import pandas as pd
try:
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset
    import torchvision.transforms as T
except ImportError:
    torch = None
    nn = None
    Dataset = object
    T = None

CLASSES = ["ACA", "N", "SCC"]
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


def get_lc25000_root(preferred_root=""):
    """Finds LC25000 root directory containing lung_aca, lung_n, lung_scc.
    Prioritizes fast 256px cache if available. Robust to any workspace folder variations.
    """
    candidates = [
        preferred_root,
        "cache_root.txt",
        "data/cache_lc25000_256/",
        "lc25000_root.txt",
        "data/LC25000/",
        "LC25000/",
        "data/lung_image_sets/",
        "lung_image_sets/",
        "data/lung_colon_image_set/lung_image_sets/",
        "lung_colon_image_set/lung_image_sets/",
        "/workspace/data/cache_lc25000_256/",
        "/workspace/data/LC25000/",
        "/workspace/LC25000/",
        "../data/LC25000/",
        "../LC25000/",
        "./"
    ]
    for c in candidates:
        if not c:
            continue
        p = c
        if c.endswith(".txt") and os.path.exists(c):
            try:
                with open(c, "r", encoding="utf-8") as f:
                    p = f.read().strip()
            except Exception:
                pass
        if p and os.path.isdir(os.path.join(p, "lung_aca")):
            resolved = p if p.endswith(("/", "\\")) else p + "/"
            return resolved

    # Dynamic fallback scan across workspace and parent directories
    search_roots = [".", "..", "data", "/workspace"]
    for base in search_roots:
        if not os.path.isdir(base):
            continue
        for root, dirs, _ in os.walk(base):
            if "lung_aca" in dirs:
                resolved = root if root.endswith(("/", "\\")) else root + "/"
                with open("lc25000_root.txt", "w", encoding="utf-8") as f:
                    f.write(resolved)
                return resolved
            # Limit depth for performance
            depth = root.count(os.sep) - base.count(os.sep)
            if depth >= 4:
                dirs.clear()

    return preferred_root if (preferred_root and preferred_root.endswith(("/", "\\"))) else (preferred_root + "/" if preferred_root else "")


def get_lunghist_root(preferred_root=""):
    """Finds LungHist700 root directory containing aca_bd.
    Handles all folder variations: data/LungHist700, data/images, LungHist700, etc.
    """
    candidates = [
        preferred_root,
        "lunghist_root.txt",
        "data/LungHist700/",
        "data/images/",
        "images/",
        "LungHist700/",
        "data/LungHist700/LungHist700/",
        "data/LungHist700/data/images/",
        "data/LungHist700/images/",
        "/workspace/data/LungHist700/",
        "/workspace/data/images/",
        "/workspace/LungHist700/",
        "../LungHist700/",
        "../data/LungHist700/",
        "data/",
        "./"
    ]
    for c in candidates:
        if not c:
            continue
        p = c
        if c.endswith(".txt") and os.path.exists(c):
            try:
                with open(c, "r", encoding="utf-8") as f:
                    p = f.read().strip()
            except Exception:
                pass
        if p and os.path.isdir(os.path.join(p, "aca_bd")):
            resolved = p if p.endswith(("/", "\\")) else p + "/"
            return resolved

    # Dynamic fallback scan across workspace and parent directories
    search_roots = [".", "..", "data", "/workspace"]
    for base in search_roots:
        if not os.path.isdir(base):
            continue
        for root, dirs, _ in os.walk(base):
            if "aca_bd" in dirs:
                resolved = root if root.endswith(("/", "\\")) else root + "/"
                with open("lunghist_root.txt", "w", encoding="utf-8") as f:
                    f.write(resolved)
                return resolved
            depth = root.count(os.sep) - base.count(os.sep)
            if depth >= 4:
                dirs.clear()

    return preferred_root if (preferred_root and preferred_root.endswith(("/", "\\"))) else (preferred_root + "/" if preferred_root else "")


def build_model(arch, num_classes=3, pretrained=False, freeze_vit_blocks=True):
    """Must mirror Section 3.3 of the manuscript."""
    if arch == "resnet50":
        import torchvision
        w = torchvision.models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        m = torchvision.models.resnet50(weights=w)
        m.fc = nn.Sequential(nn.Dropout(0.3), nn.Linear(2048, num_classes))
        return m
    if arch in ["vit_s16", "vit_s"]:
        import timm
        m = timm.create_model(
            "vit_small_patch16_224.augreg_in21k_ft_in1k",
            pretrained=pretrained,
            num_classes=num_classes
        )
        if freeze_vit_blocks:
            # Freeze encoder blocks 0 and 1 as specified in Section 3.3
            for name, param in m.named_parameters():
                if any(name.startswith(f"blocks.{b}.") for b in [0, 1]):
                    param.requires_grad = False
        return m
    raise ValueError(f"Unknown architecture: {arch}")


def count_parameters(model):
    """Counts total and trainable parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def resolve_checkpoint_path(path):
    """Finds the actual checkpoint path regardless of folder naming variations."""
    if not path:
        return path
    if os.path.exists(path):
        return path

    path_norm = path.replace("\\", "/")
    if os.path.exists(path_norm):
        return path_norm

    replacements = [
        ("vit_s_main", "vit_s16_main"),
        ("vit_s16_main", "vit_s_main"),
        ("vit_s16", "vit_s"),
        ("vit_s", "vit_s16"),
        ("resnet50_main", "resnet50_adam"),
        ("resnet50_adam", "resnet50_main"),
        ("resnet50_sgdm", "resnet50_sgdm_1e4"),
        ("resnet50_sgdm_1e4", "resnet50_sgdm"),
        ("vit_s_sgdm", "vit_s16_sgdm"),
        ("vit_s16_sgdm", "vit_s_sgdm"),
        ("clean_resnet50", "clean_r50"),
        ("clean_r50", "clean_resnet50"),
        ("clean_vit_s16", "clean_vit_s"),
        ("clean_vit_s", "clean_vit_s16"),
        ("runs/", "results/"),
        ("results/", "runs/"),
        ("runs/", "")
    ]
    for old_s, new_s in replacements:
        cand = path_norm.replace(old_s, new_s)
        if os.path.exists(cand):
            return cand
        cand_orig = path.replace(old_s, new_s)
        if os.path.exists(cand_orig):
            return cand_orig

    # Recursive fallback search across workspace
    base_name = os.path.basename(path)
    parent_target = os.path.basename(os.path.dirname(path))
    for s_dir in ["runs", "results", "/workspace", "."]:
        if os.path.isdir(s_dir):
            for r, _, fs in os.walk(s_dir):
                if base_name in fs:
                    curr_parent = os.path.basename(r)
                    if not parent_target or parent_target in curr_parent or curr_parent in parent_target:
                        return os.path.join(r, base_name)
    return path


def load_checkpoint(model, path, device="cpu"):
    real_path = resolve_checkpoint_path(path)
    if not os.path.exists(real_path):
        raise FileNotFoundError(f"Checkpoint not found at '{path}' (or any alias). Checked: {real_path}")
    sd = torch.load(real_path, map_location=device)
    if isinstance(sd, dict):
        for k in ("model_state_dict", "state_dict", "model"):
            if k in sd and isinstance(sd[k], dict):
                sd = sd[k]
                break
    sd = {k.replace("module.", ""): v for k, v in sd.items()}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing or unexpected:
        print(f"[warn] {real_path}: missing={len(missing)} unexpected={len(unexpected)}")
    return model


def train_transform(img_size=224):
    """Training data augmentation mirroring Section 3.3."""
    return T.Compose([
        T.Resize((img_size, img_size)),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomRotation(degrees=20),
        T.RandomAffine(degrees=0, scale=(0.9, 1.1), shear=10),
        T.ColorJitter(brightness=0.2, contrast=0.2),
        T.ToTensor(),
        T.Normalize(MEAN, STD)
    ])


def eval_transform(mode="resize"):
    """mode='resize': direct resize to 224x224 (as in the original zero-shot test).
    mode='crop768': centre crop <=768 px then resize (as in the fine-tuning experiment)."""
    if mode == "crop768":
        return T.Compose([
            T.Lambda(lambda im: T.CenterCrop(min(768, *im.size))(im)),
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(MEAN, STD)
        ])
    return T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(MEAN, STD)
    ])


def label_to_int(v):
    if isinstance(v, (int, np.integer)):
        return int(v)
    v = str(v).strip().upper()
    alias = {"NORMAL": "N", "BENIGN": "N", "LUNG_N": "N", "LUNG_ACA": "ACA", "LUNG_SCC": "SCC"}
    return CLASSES.index(alias.get(v, v))


class ImageList(Dataset):
    def __init__(self, csv, transform, root=""):
        self.df = pd.read_csv(csv) if isinstance(csv, str) else csv.reset_index(drop=True)
        self.t = transform
        # Automatically detect and refine dataset root if needed
        if len(self.df) > 0 and "aca_bd" in str(self.df.iloc[0]["path"]):
            self.root = get_lunghist_root(root)
        elif len(self.df) > 0 and "lung_" in str(self.df.iloc[0]["path"]):
            self.root = get_lc25000_root(root)
        else:
            self.root = root

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        clean_rel = str(r['path']).replace("\\", "/").lstrip("/")
        img_path = os.path.join(self.root, clean_rel)
        if not os.path.exists(img_path):
            if os.path.exists(clean_rel):
                img_path = clean_rel
            elif os.path.exists(str(r['path'])):
                img_path = str(r['path'])
            else:
                raise FileNotFoundError(f"Image not found at {img_path} (root: {self.root})")
        im = Image.open(img_path).convert("RGB")
        return self.t(im), label_to_int(r["label"]), i


def wilson(k, n, z=1.959964):
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return 100 * (c - h), 100 * (c + h)


def metrics(y, prob):
    from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score,
                                 confusion_matrix)
    pred = prob.argmax(1)
    out = dict(
        n=len(y),
        correct=int((pred == y).sum()),
        acc=100 * accuracy_score(y, pred),
        bal_acc=100 * balanced_accuracy_score(y, pred),
        macro_f1=float(f1_score(y, pred, average="macro"))
    )
    try:
        out["macro_auc"] = float(roc_auc_score(y, prob, multi_class="ovr", average="macro"))
    except ValueError:
        out["macro_auc"] = float("nan")
    out["acc_ci"] = [float(x) for x in wilson(out["correct"], out["n"])]
    out["f1_per_class"] = [float(x) for x in f1_score(y, pred, average=None, labels=[0, 1, 2]).round(4)]
    out["confusion"] = confusion_matrix(y, pred, labels=[0, 1, 2]).tolist()
    return out


def cluster_bootstrap(y, pred, groups, stat=lambda y, p: 100 * (y == p).mean(), B=2000, seed=0):
    """Patient-level cluster bootstrap 95% CI (as in Section 3.7)."""
    rng = np.random.default_rng(seed)
    ug = np.unique(groups)
    idx = {g: np.where(groups == g)[0] for g in ug}
    vals = []
    for _ in range(B):
        s = np.concatenate([idx[g] for g in rng.choice(ug, len(ug), replace=True)])
        vals.append(stat(y[s], pred[s]))
    return [float(x) for x in np.percentile(vals, [2.5, 97.5])]
