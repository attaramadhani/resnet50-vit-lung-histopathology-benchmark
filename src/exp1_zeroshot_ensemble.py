"""Exp 1 - Zero-shot test of the MAIN benchmark ensembles (scenario 1 ResNet50, scenario 6 ViT-S/16)
on all 691 LungHist700 images. Replaces the 1,800-image / 4-epoch / CPU model (manuscript M1).

Example:
  python exp1_zeroshot_ensemble.py --arch resnet50 \
     --ckpts runs/resnet50_main/fold{0..4}.pt --csv lunghist700.csv --root data/LungHist700/ \
     --preprocess resize --out exp1_resnet50_main
"""
import argparse, json
import numpy as np, pandas as pd, torch
from torch.utils.data import DataLoader
from common import build_model, load_checkpoint, eval_transform, ImageList, metrics, cluster_bootstrap, label_to_int

ap = argparse.ArgumentParser()
ap.add_argument("--arch", required=True, choices=["resnet50", "vit_s16"])
ap.add_argument("--ckpts", nargs="+", required=True, help="the five fold checkpoints")
ap.add_argument("--csv", required=True); ap.add_argument("--root", default="")
ap.add_argument("--preprocess", default="resize", choices=["resize", "crop768"])
ap.add_argument("--bs", type=int, default=32); ap.add_argument("--out", required=True)
a = ap.parse_args()
dev = "cuda" if torch.cuda.is_available() else "cpu"

ds = ImageList(a.csv, eval_transform(a.preprocess), a.root)
dl = DataLoader(ds, batch_size=a.bs, num_workers=4)
probs = []
for c in a.ckpts:
    m = load_checkpoint(build_model(a.arch), c, dev).to(dev).eval()
    P = []
    with torch.no_grad(), torch.autocast(dev, enabled=(dev == "cuda")):
        for x, _, _ in dl:
            P.append(torch.softmax(m(x.to(dev)).float(), 1).cpu())
    probs.append(torch.cat(P).numpy())
prob = np.mean(probs, 0)                       # soft voting, as in Section 3.5
y = ds.df["label"].map(label_to_int).to_numpy()
res = metrics(y, prob)
pred = prob.argmax(1)
res["recall_per_class"] = [float(100 * (pred[y == k] == k).mean()) for k in range(3)]
if "patient" in ds.df:
    res["acc_cluster_ci"] = cluster_bootstrap(y, pred, ds.df["patient"].to_numpy())
if "magnification" in ds.df:
    res["acc_by_mag"] = {str(g): float(100 * (pred[ds.df.magnification == g] == y[ds.df.magnification == g]).mean())
                         for g in ds.df.magnification.unique()}
out = ds.df.copy(); out[["p_ACA", "p_N", "p_SCC"]] = prob; out["pred"] = pred
out.to_csv(a.out + "_preds.csv", index=False)
json.dump(res, open(a.out + "_metrics.json", "w"), indent=2)
print(json.dumps(res, indent=2))
