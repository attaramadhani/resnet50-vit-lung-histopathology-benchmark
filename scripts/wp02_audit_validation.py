"""WP02 - Validation of the rotation-invariant near-duplicate audit (Reviewers 2.3, 3.2, 5.5a, 6.1).

LC25000 has no source identifiers, so the audit's recall cannot be measured on LC25000 itself.
Here we BUILD a pool with KNOWN siblings and run the same audit on it:

  1. Pick S source images per class from LC25000, each from a DIFFERENT audit group
     (data_splits/exp4_groups.csv), seed 42.
  2. For each source make M synthetic siblings with the documented LC25000 augmentation:
     random rotation in [-25, +25] degrees (rotate, then centre-crop the largest valid square and
     resize back to the original size, so no black corners) and random horizontal / vertical flips.
     The source itself is kept too, so each source family has M + 1 images.
  3. Run EXACTLY the exp4 audit on this pool: ResNet50 (ImageNet-1K V2) embeddings averaged over the
     8 dihedral transforms, class-wise k-NN on cosine similarity, ORB (1500 keypoints, 384 px grey)
     + ratio test 0.75 + RANSAC partial-affine (reprojection 5 px), union-find on linked pairs.
  4. Compare predicted groups with the true families:
       pair recall  = true sibling pairs that end in the same predicted group / all true sibling pairs
       pair precision = predicted same-group pairs that are true siblings / all predicted same-group pairs
       FNR = 1 - recall; also the number of predicted groups vs S, and the share of families fully recovered.
  5. Sensitivity grid: cosine threshold x k x minimum inliers (default 0.90 / 30 / 25 is marked).

Caveat to report: two "different" LC25000 groups may still share a source (the audit is known to miss
links), so a few false positives can be real siblings -> precision is a LOWER bound.

Usage (GPU recommended, about 5-10 min on an RTX 4090 with defaults):
  python wp02_audit_validation.py --root /workspace/data/LC25000/ --splits data_splits --out results_wp02
  (--root must hold the ORIGINAL 768x768 images in lung_aca/, lung_n/, lung_scc/)

Outputs in --out: wp02_grid.csv, wp02_grid.md (default row in bold), wp02_pairs.csv, wp02_pool.csv
"""
import argparse
import itertools
import math
import os
import random

import cv2
import numpy as np
import pandas as pd
import torch
import torchvision
import torchvision.transforms as T
from PIL import Image

CLASSES = ["ACA", "N", "SCC"]
ALIAS = {"LUNG_ACA": "ACA", "LUNG_N": "N", "LUNG_SCC": "SCC"}
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


def lab(v):
    v = str(v).strip().upper()
    return CLASSES.index(ALIAS.get(v, v))


def rotate_crop(im, angle):
    """Rotate by angle (deg) and crop the largest axis-aligned square without empty corners."""
    w, h = im.size
    r = im.rotate(angle, resample=Image.BICUBIC, expand=False)
    a = math.radians(abs(angle))
    side = w / (math.cos(a) + math.sin(a))
    left = (w - side) / 2
    return r.crop((left, left, left + side, left + side)).resize((w, h), Image.BICUBIC)


def make_pool(groups, root, S, M, out_dir, seed):
    rng = random.Random(seed)
    os.makedirs(os.path.join(out_dir, "pool"), exist_ok=True)
    rows = []
    for c in CLASSES:
        g = groups[groups.label.map(lab) == CLASSES.index(c)]
        one_per_group = g.groupby("group").head(1).sample(frac=1.0, random_state=seed).head(S)
        for fam, p in enumerate(one_per_group.path):
            src = Image.open(os.path.join(root, p)).convert("RGB")
            for s in range(M + 1):
                im = src if s == 0 else rotate_crop(src, rng.uniform(-25, 25))
                if s > 0 and rng.random() < 0.5:
                    im = im.transpose(Image.FLIP_LEFT_RIGHT)
                if s > 0 and rng.random() < 0.5:
                    im = im.transpose(Image.FLIP_TOP_BOTTOM)
                name = f"pool/{c}_{fam:03d}_{s:02d}.png"
                im.save(os.path.join(out_dir, name))
                rows.append(dict(path=name, label=c, family=f"{c}_{fam}", source=p))
    return pd.DataFrame(rows)


def embeddings(df, base, dev, no_pretrained=False):
    w = None if no_pretrained else torchvision.models.ResNet50_Weights.IMAGENET1K_V2
    enc = torchvision.models.resnet50(weights=w)
    enc.fc = torch.nn.Identity()
    enc = enc.to(dev).eval()
    tf = T.Compose([T.Resize((224, 224)), T.ToTensor(), T.Normalize(MEAN, STD)])
    E = []
    with torch.no_grad():
        for i in range(0, len(df), 64):
            x = torch.stack([tf(Image.open(os.path.join(base, p)).convert("RGB")) for p in df.path[i:i + 64]]).to(dev)
            f = 0
            for k in range(4):
                r = torch.rot90(x, k, (2, 3))
                f = f + enc(r) + enc(torch.flip(r, (3,)))
            E.append(torch.nn.functional.normalize(f, dim=1).cpu())
    return torch.cat(E)


def orb_inliers(df, base, pairs):
    orb = cv2.ORB_create(1500)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    cache = {}

    def kp(i):
        if i not in cache:
            im = cv2.imread(os.path.join(base, df.path[i]), cv2.IMREAD_GRAYSCALE)
            cache[i] = orb.detectAndCompute(cv2.resize(im, (384, 384)), None)
        return cache[i]

    out = []
    for i, j in pairs:
        (k1, d1), (k2, d2) = kp(i), kp(j)
        if d1 is None or d2 is None or len(k1) < 10 or len(k2) < 10:
            out.append(0)
            continue
        good = [m for m, n in (p for p in bf.knnMatch(d1, d2, k=2) if len(p) == 2) if m.distance < 0.75 * n.distance]
        if len(good) < 10:
            out.append(0)
            continue
        src = np.float32([k1[m.queryIdx].pt for m in good])
        dst = np.float32([k2[m.trainIdx].pt for m in good])
        _, mask = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC, ransacReprojThreshold=5.0)
        out.append(int(mask.sum()) if mask is not None else 0)
    return out


def groups_from(n, links):
    parent = list(range(n))

    def find(u):
        while parent[u] != u:
            parent[u] = parent[parent[u]]
            u = parent[u]
        return u

    for i, j in links:
        parent[find(i)] = find(j)
    return np.array([find(i) for i in range(n)])


def evaluate(df, pred_group):
    fam = df.family.values
    tp = fp = fn = 0
    for i, j in itertools.combinations(range(len(df)), 2):
        same_true = fam[i] == fam[j]
        same_pred = pred_group[i] == pred_group[j]
        tp += same_true and same_pred
        fp += (not same_true) and same_pred
        fn += same_true and (not same_pred)
    rec = tp / (tp + fn) if tp + fn else float("nan")
    prec = tp / (tp + fp) if tp + fp else float("nan")
    full = np.mean([len(set(pred_group[fam == f])) == 1 for f in np.unique(fam)])
    return dict(pair_recall=round(100 * rec, 2), pair_precision=round(100 * prec, 2), fnr=round(100 * (1 - rec), 2),
                n_pred_groups=int(len(np.unique(pred_group))), n_true_families=int(len(np.unique(fam))),
                families_fully_recovered=round(100 * full, 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="", help="Original 768x768 LC25000 root (auto-resolved if empty)")
    ap.add_argument("--splits", default="data_splits")
    ap.add_argument("--out", default="results_rev1/WP02")
    ap.add_argument("--sources-per-class", type=int, default=50)
    ap.add_argument("--siblings", type=int, default=9, help="synthetic siblings per source (family size = this + 1)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-pretrained", action="store_true", help="debug only")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    src_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src")
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)
    from common import get_lc25000_root, resolve_split_csv
    raw_root = get_lc25000_root(a.root, require_raw=True)
    if not raw_root or not os.path.isdir(os.path.join(raw_root, "lung_aca")):
        raise FileNotFoundError(f"Original 768x768 LC25000 root not found: '{a.root}' / '{raw_root}'")
    groups_csv = os.path.join(a.splits, "exp4_groups.csv")
    if not os.path.exists(groups_csv):
        groups_csv = resolve_split_csv("exp4_groups.csv")

    groups = pd.read_csv(groups_csv)
    pool = make_pool(groups, raw_root, a.sources_per_class, a.siblings, a.out, a.seed)
    pool.to_csv(os.path.join(a.out, "wp02_pool.csv"), index=False)
    print(f"Pool: {len(pool)} images, {pool.family.nunique()} families")

    E = embeddings(pool, a.out, dev, a.no_pretrained)
    y = pool.label.map(lab).values
    K_MAX, SIM_MIN = 50, 0.80
    cand = []  # (i, j, cos, rank of j in i's list)
    for c in range(3):
        idx = np.where(y == c)[0]
        S = E[idx] @ E[idx].T
        S.fill_diagonal_(-1)
        top = S.topk(min(K_MAX, len(idx) - 1), dim=1)
        for r, i in enumerate(idx):
            for rank, (s, jpos) in enumerate(zip(top.values[r].tolist(), top.indices[r].tolist())):
                j = idx[jpos]
                if j <= i or s < SIM_MIN:
                    continue
                cand.append((i, j, s, rank))
    print(f"Candidates (k<= {K_MAX}, cos>= {SIM_MIN}): {len(cand)}; computing ORB/RANSAC inliers...")
    inl = orb_inliers(pool, a.out, [(i, j) for i, j, _, _ in cand])
    P = pd.DataFrame(cand, columns=["i", "j", "cos", "rank"])
    P["inliers"] = inl
    P["true_sibling"] = pool.family.values[P.i] == pool.family.values[P.j]
    P.to_csv(os.path.join(a.out, "wp02_pairs.csv"), index=False)

    rows = []
    for sim, k, mi in itertools.product([0.80, 0.85, 0.90, 0.95], [10, 20, 30, 50], [10, 15, 25, 40]):
        L = P[(P.cos >= sim) & (P["rank"] < k) & (P.inliers >= mi)]
        g = groups_from(len(pool), zip(L.i, L.j))
        r = evaluate(pool, g)
        r.update(cosine=sim, k=k, min_inliers=mi, default=(sim == 0.90 and k == 30 and mi == 25))
        rows.append(r)
    R = pd.DataFrame(rows)[["cosine", "k", "min_inliers", "default", "pair_recall", "pair_precision", "fnr",
                            "n_pred_groups", "n_true_families", "families_fully_recovered"]]
    R.to_csv(os.path.join(a.out, "wp02_grid.csv"), index=False)
    with open(os.path.join(a.out, "wp02_grid.md"), "w") as f:
        f.write("| Cosine | k | Min. inliers | Pair recall (%) | Pair precision (%) | FNR (%) | Groups / families | Families fully recovered (%) |\n")
        f.write("| --- | --- | --- | --- | --- | --- | --- | --- |\n")
        for _, r in R.iterrows():
            b = "**" if r["default"] else ""
            f.write(f"| {b}{r.cosine:.2f}{b} | {b}{r.k}{b} | {b}{r.min_inliers}{b} | {b}{r.pair_recall}{b} | "
                    f"{b}{r.pair_precision}{b} | {b}{r.fnr}{b} | {r.n_pred_groups} / {r.n_true_families} | {r.families_fully_recovered} |\n")
    d = R[R["default"]].iloc[0]
    print(f"\nDEFAULT (cos 0.90, k 30, inliers 25): recall {d.pair_recall}%, precision {d.pair_precision}%, "
          f"FNR {d.fnr}%, {d.n_pred_groups} groups for {d.n_true_families} families, "
          f"{d.families_fully_recovered}% families fully recovered")


if __name__ == "__main__":
    main()
