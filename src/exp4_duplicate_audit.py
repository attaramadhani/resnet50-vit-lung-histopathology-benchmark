"""Exp 4 - Rotation-invariant near-duplicate audit of the LC25000 lung subset (manuscript M3, Section 3.2).

LC25000 was augmented from 250 source images per class with arbitrary rotations and flips, so pHash cannot
find the relations. This script:
  1. embeds every image with an ImageNet ResNet50, averaging features over the 8 dihedral transforms
     (rotation/flip-robust descriptor);
  2. within each class, takes the k nearest neighbours by cosine similarity as candidate pairs;
  3. verifies each candidate pair geometrically with ORB keypoints + RANSAC (rotation invariant);
  4. merges verified pairs with union-find into source groups and writes group ids.
Expected if it works: about 250 groups per class of about 20 images each.
Then:   python exp4_duplicate_audit.py ... --make-split  -> grouped dev/test split + StratifiedGroupKFold folds.

  python exp4_duplicate_audit.py --csv lc25000_lung_all.csv --root data/LC25000/ --out exp4
"""
import argparse, json
import numpy as np, pandas as pd, torch, cv2
from torch.utils.data import DataLoader
from common import build_model, eval_transform, ImageList, label_to_int

ap = argparse.ArgumentParser()
ap.add_argument("--csv", required=True); ap.add_argument("--root", default="")
ap.add_argument("--k", type=int, default=30); ap.add_argument("--sim", type=float, default=0.90)
ap.add_argument("--min-inliers", type=int, default=25); ap.add_argument("--out", required=True)
ap.add_argument("--make-split", action="store_true"); ap.add_argument("--test-frac", type=float, default=0.2)
ap.add_argument("--max-images", type=int, default=0, help="debug: limit images")
ap.add_argument("--no-pretrained", action="store_true", help="debug only")
a = ap.parse_args()
dev = "cuda" if torch.cuda.is_available() else "cpu"

df = pd.read_csv(a.csv)
if a.max_images: df = df.groupby("label", group_keys=False).head(a.max_images // 3)
df = df.reset_index(drop=True); df["y"] = df.label.map(label_to_int)

# ---- 1. dihedral-averaged embeddings
import torchvision
enc = torchvision.models.resnet50(weights=None if a.no_pretrained else torchvision.models.ResNet50_Weights.IMAGENET1K_V2)
enc.fc = torch.nn.Identity(); enc = enc.to(dev).eval()
dl = DataLoader(ImageList(df, eval_transform("resize"), a.root), batch_size=64, num_workers=4)
E = []
with torch.no_grad():
    for x, _, _ in dl:
        x = x.to(dev); f = 0
        for k in range(4):
            r = torch.rot90(x, k, (2, 3))
            f = f + enc(r) + enc(torch.flip(r, (3,)))
        E.append(torch.nn.functional.normalize(f, dim=1).cpu())
E = torch.cat(E)

# ---- 2+3. candidates and ORB/RANSAC verification
orb = cv2.ORB_create(1500); bf = cv2.BFMatcher(cv2.NORM_HAMMING)
_kp = {}
def kp(i):
    if i not in _kp:
        im = cv2.imread(f"{a.root}{df.path[i]}", cv2.IMREAD_GRAYSCALE)
        im = cv2.resize(im, (384, 384))
        _kp[i] = orb.detectAndCompute(im, None)
        if len(_kp) > 4000: _kp.pop(next(iter(_kp)))
    return _kp[i]

def inliers(i, j):
    (k1, d1), (k2, d2) = kp(i), kp(j)
    if d1 is None or d2 is None or len(k1) < 10 or len(k2) < 10: return 0
    good = [m for m, n in (p for p in bf.knnMatch(d1, d2, k=2) if len(p) == 2) if m.distance < 0.75 * n.distance]
    if len(good) < 10: return 0
    src = np.float32([k1[m.queryIdx].pt for m in good]); dst = np.float32([k2[m.trainIdx].pt for m in good])
    M, mask = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC, ransacReprojThreshold=5.0)
    return int(mask.sum()) if mask is not None else 0

parent = list(range(len(df)))
def find(u):
    while parent[u] != u: parent[u] = parent[parent[u]]; u = parent[u]
    return u
pairs = []
for c in sorted(df.y.unique()):
    idx = np.where(df.y == c)[0]; S = E[idx] @ E[idx].T
    S.fill_diagonal_(-1)
    top = S.topk(min(a.k, len(idx) - 1), dim=1)
    for r, i in enumerate(idx):
        for s, jpos in zip(top.values[r].tolist(), top.indices[r].tolist()):
            j = idx[jpos]
            if j <= i or s < a.sim: continue
            n = inliers(i, j)
            pairs.append((i, j, s, n))
            if n >= a.min_inliers: parent[find(i)] = find(j)
    print(f"class {c}: done")
df["group"] = [find(i) for i in range(len(df))]
pd.DataFrame(pairs, columns=["i", "j", "cos", "inliers"]).to_csv(a.out + "_pairs.csv", index=False)
sizes = df.groupby(["label", "group"]).size()
stats = {str(l): dict(n_groups=int(s.size), median_size=float(s.median()), max_size=int(s.max()),
                      singletons=int((s == 1).sum())) for l, s in sizes.groupby(level=0)}
print(json.dumps(stats, indent=2)); json.dump(stats, open(a.out + "_stats.json", "w"), indent=2)
df.drop(columns="y").to_csv(a.out + "_groups.csv", index=False)

# ---- 4. grouped split for re-training scenarios 1 and 6
if a.make_split:
    from sklearn.model_selection import StratifiedGroupKFold
    sg = StratifiedGroupKFold(n_splits=round(1 / a.test_frac), shuffle=True, random_state=42)
    dev_idx, test_idx = next(sg.split(df, df.y, df.group))
    df["partition"] = "dev"; df.loc[test_idx, "partition"] = "test"; df["fold"] = -1
    d = df.loc[dev_idx]
    for f, (_, va) in enumerate(StratifiedGroupKFold(5, shuffle=True, random_state=42).split(d, d.y, d.group)):
        df.loc[d.index[va], "fold"] = f
    assert not set(df[df.partition == "dev"].group) & set(df[df.partition == "test"].group)
    df.drop(columns="y").to_csv(a.out + "_grouped_split.csv", index=False)
    print(df.groupby(["partition", "label"]).size())
