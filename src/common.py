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
from PIL import Image
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


def resolve_split_csv(path):
    """Resolve split CSV across cwd, data_splits/, parent dirs, or /workspace."""
    if not path:
        return path
    if os.path.exists(path):
        return path
    base = os.path.basename(path)
    candidates = [
        os.path.join("data_splits", base),
        os.path.join("..", "data_splits", base),
        os.path.join("/workspace", "resnet50-vit-lung-histopathology-benchmark", "data_splits", base),
    ]
    for cand in candidates:
        if os.path.exists(cand):
            return cand
    return path


def _is_full_lc25000_dir(folder, require_raw=False):
    sample_dir = os.path.join(folder, "lung_aca")
    if not os.path.isdir(sample_dir):
        return False
    try:
        imgs = [fn for fn in os.listdir(sample_dir) if fn.lower().endswith((".jpeg", ".jpg", ".png"))]
        if len(imgs) < 4000:
            return False
        if require_raw:
            with Image.open(os.path.join(sample_dir, imgs[0])) as im:
                return max(im.size) >= 512
        return True
    except Exception:
        return False


def get_lc25000_root(preferred_root="", require_raw=False):
    """Finds LC25000 root directory containing lung_aca, lung_n, lung_scc.
    When require_raw=True (WP02/WP06f), rejects 256px caches and resolves 768x768 originals.
    """
    raw_candidates = [
        preferred_root,
        os.environ.get("LC25000_RAW_ROOT", "") if require_raw else os.environ.get("LC25000_CACHE_ROOT", ""),
        "lc25000_raw_root.txt",
        "data/LC25000_raw/",
        "data/LC25000/",
        "LC25000/",
        "data/lung_image_sets/",
        "lung_image_sets/",
        "data/lung_colon_image_set/lung_image_sets/",
        "lung_colon_image_set/lung_image_sets/",
        "/workspace/data/LC25000_raw/",
        "/workspace/data/LC25000/",
        "/workspace/LC25000/",
        "../data/LC25000/",
        "../LC25000/",
    ]
    if require_raw:
        candidates = raw_candidates
    else:
        candidates = [
            preferred_root,
            os.environ.get("LC25000_CACHE_ROOT", ""),
            "cache_root.txt",
            "data/cache_lc25000_256/",
            "/workspace/data/cache_lc25000_256/",
            "lc25000_root.txt",
        ] + raw_candidates + ["./"]

    def _matches(folder):
        if not folder or not os.path.isdir(os.path.join(folder, "lung_aca")):
            return False
        return _is_full_lc25000_dir(folder, require_raw=require_raw)

    for c in candidates:
        if not c:
            continue
        p = c
        if c.endswith(".txt") and os.path.exists(c):
            try:
                with open(c, "r", encoding="utf-8") as f:
                    p = f.read().strip()
            except Exception:
                continue
        if _matches(p):
            return p if p.endswith(("/", "\\")) else p + "/"

    # Dynamic fallback scan across workspace, data, kagglehub cache, and parent directories
    search_roots = [".", "..", "data", "/workspace", os.environ.get("KAGGLEHUB_CACHE_DIR", ""), os.path.expanduser("~/.cache/kagglehub")]
    for base in search_roots:
        if not base or not os.path.isdir(base):
            continue
        for root, dirs, _ in os.walk(base):
            if "lung_aca" in dirs and _matches(root):
                resolved = root if root.endswith(("/", "\\")) else root + "/"
                target_txt = "lc25000_raw_root.txt" if require_raw else "lc25000_root.txt"
                try:
                    with open(target_txt, "w", encoding="utf-8") as f:
                        f.write(resolved)
                except Exception:
                    pass
                return resolved
            depth = root.count(os.sep) - base.count(os.sep)
            if depth >= 6:
                dirs.clear()

    return preferred_root if (preferred_root and preferred_root.endswith(("/", "\\"))) else (preferred_root + "/" if preferred_root else "")


def get_lunghist_root(preferred_root=""):
    """Finds LungHist700 root directory containing aca_bd.
    Handles all folder variations: data/LungHist700, data/images, LungHist700, /workspace/data/LungHist700, etc.
    """
    candidates = [
        preferred_root,
        os.environ.get("LUNGHIST700_ROOT", ""),
        "lunghist_root.txt",
        "data/LungHist700_images/",
        "data/LungHist700/",
        "data/images/",
        "images/",
        "LungHist700/",
        "data/LungHist700/LungHist700/",
        "data/LungHist700/data/images/",
        "data/LungHist700/images/",
        "/workspace/data/LungHist700_images/",
        "/workspace/data/LungHist700/",
        "/workspace/data/LungHist700/data/images/",
        "/workspace/data/images/",
        "/workspace/LungHist700/",
        "../LungHist700/",
        "../data/LungHist700/",
        "data/",
        "./"
    ]
    def _is_valid_lh(folder):
        aca_dir = os.path.join(folder, "aca_bd")
        if not folder or not os.path.isdir(aca_dir):
            return False
        try:
            return len([x for x in os.listdir(aca_dir) if x.lower().endswith((".jpg", ".jpeg", ".png"))]) >= 50
        except Exception:
            return False

    for c in candidates:
        if not c:
            continue
        p = c
        if c.endswith(".txt") and os.path.exists(c):
            try:
                with open(c, "r", encoding="utf-8") as f:
                    p = f.read().strip()
            except Exception:
                continue
        if _is_valid_lh(p):
            return p if p.endswith(("/", "\\")) else p + "/"

    search_roots = [".", "..", "data", "/workspace"]
    for base in search_roots:
        if not os.path.isdir(base):
            continue
        for root, dirs, _ in os.walk(base):
            if "aca_bd" in dirs and _is_valid_lh(root):
                resolved = root if root.endswith(("/", "\\")) else root + "/"
                try:
                    with open("lunghist_root.txt", "w", encoding="utf-8") as f:
                        f.write(resolved)
                except Exception:
                    pass
                return resolved
            depth = root.count(os.sep) - base.count(os.sep)
            if depth >= 5:
                dirs.clear()

    return preferred_root if (preferred_root and preferred_root.endswith(("/", "\\"))) else (preferred_root + "/" if preferred_root else "")


def build_model(arch, num_classes=3, pretrained=False, freeze_vit_blocks=True, head_dropout=None):
    """Must mirror Section 3.3 of the manuscript.
    When head_dropout is None, keeps v2 defaults (ResNet50 Dropout(0.3), ViT drop_rate=0.0).
    """
    if arch == "resnet50":
        import torchvision
        w = torchvision.models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        m = torchvision.models.resnet50(weights=w)
        p_drop = 0.3 if head_dropout is None else float(head_dropout)
        if p_drop > 0.0:
            m.fc = nn.Sequential(nn.Dropout(p_drop), nn.Linear(2048, num_classes))
        else:
            m.fc = nn.Linear(2048, num_classes)
        return m
    if arch in ["vit_s16", "vit_s"]:
        import timm
        p_drop = 0.0 if head_dropout is None else float(head_dropout)
        m = timm.create_model(
            "vit_small_patch16_224.augreg_in21k_ft_in1k",
            pretrained=pretrained,
            num_classes=num_classes,
            drop_rate=p_drop,
        )
        if freeze_vit_blocks:
            # Freeze encoder blocks 0 and 1 as specified in Section 3.3
            for name, param in m.named_parameters():
                if any(name.startswith(f"blocks.{b}.") for b in [0, 1]):
                    param.requires_grad = False
        return m
    if arch == "deit_s16":
        import timm
        p_drop = 0.0 if head_dropout is None else float(head_dropout)
        m = timm.create_model(
            "deit_small_patch16_224.fb_in1k",
            pretrained=pretrained,
            num_classes=num_classes,
            drop_rate=p_drop,
        )
        if freeze_vit_blocks:
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
    """Finds actual checkpoint path across folder aliases (old_ckpts, runs, results) and .pt/.pth extensions."""
    if not path:
        return path

    def _check_ext(p):
        if os.path.exists(p):
            return p
        if p.endswith(".pt") and os.path.exists(p + "h"):
            return p + "h"
        if p.endswith(".pth") and os.path.exists(p[:-1]):
            return p[:-1]
        return None

    hit = _check_ext(path)
    if hit:
        return hit

    path_norm = path.replace("\\", "/")
    hit = _check_ext(path_norm)
    if hit:
        return hit

    replacements = [
        ("old_ckpts/runs/", "runs/"),
        ("old_ckpts/runs/", "old_ckpts/"),
        ("old_ckpts/", "old_ckpts/runs/"),
        ("old_ckpts/", "runs/"),
        ("runs/", "old_ckpts/runs/"),
        ("runs/", "old_ckpts/"),
        ("runs/", "results/"),
        ("results/", "runs/"),
        ("runs/", ""),
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
    ]
    for old_s, new_s in replacements:
        for base_p in (path_norm, path):
            cand = base_p.replace(old_s, new_s)
            hit = _check_ext(cand)
            if hit:
                return hit

    # Recursive fallback search across workspace (.pt and .pth) with canonical parent matching
    def _canon(s):
        return (s.lower()
                .replace("vit_s16", "vit_s")
                .replace("clean_r50", "clean_resnet50")
                .replace("resnet50_adam", "resnet50_main")
                .replace("sgdm_1e4", "sgdm"))

    base_stem = os.path.splitext(os.path.basename(path_norm))[0]
    parent_target = _canon(os.path.basename(os.path.dirname(path_norm)))
    for s_dir in ["old_ckpts", "runs", "results", "/workspace", "."]:
        if os.path.isdir(s_dir):
            for r, _, fs in os.walk(s_dir):
                for fn in fs:
                    stem, ext = os.path.splitext(fn)
                    if stem == base_stem and ext.lower() in (".pt", ".pth"):
                        curr_parent = _canon(os.path.basename(r))
                        if not parent_target or parent_target == curr_parent or parent_target in curr_parent or curr_parent in parent_target:
                            return os.path.join(r, fn)
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
    prob = np.asarray(prob, dtype=np.float64)
    if np.isnan(prob).any() or np.isinf(prob).any():
        prob = np.nan_to_num(prob, nan=1.0 / 3.0, posinf=1.0, neginf=0.0)
        row_sums = np.maximum(prob.sum(axis=1, keepdims=True), 1e-8)
        prob = prob / row_sums
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
