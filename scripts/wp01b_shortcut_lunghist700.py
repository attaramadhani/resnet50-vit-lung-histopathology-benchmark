"""WP01b - Negative control for the LC25000 shortcut test, on LungHist700.

Runs the six hand-crafted feature sets of WP01 (logistic regression and random forest)
on LungHist700 with the patient-grouped 5-fold split of Section 3.7
(data_splits/folds_patient.csv). Reports out-of-fold accuracy over all 691 images
with a patient-level cluster-bootstrap 95% CI (2,000 resamples of the 45 patients).

Preprocessing matches the LungHist700 evaluation pipeline: center crop of at most
768 px, then resize to 256 x 256 (the size WP01 used for LC25000 features).
The Reinhard target is the mean Lab statistics of 300 random LungHist700 images (seed 42).

Place this file in scripts/ next to wp01_shortcut_lc25000.py. Usage, from the repo root:
  python scripts/wp01b_shortcut_lunghist700.py --root /path/to/LungHist700 --out results_rev1/WP01b
CPU only; a few minutes.

Interpretation for Section 5.4: the colour-histogram forest reached 97.83% on the
group-aware LC25000 split. A much lower value here (e.g. < 70%) supports the claim that
the colour shortcut is a property of LC25000. A value above about 85% does not.
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score, recall_score

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "src"))
from wp01_shortcut_lc25000 import features_for, rgb_to_lab, classifiers, lab, SEED  # noqa: E402
from common import cluster_bootstrap  # noqa: E402


def load(path):
    im = Image.open(path).convert("RGB")
    s = min(768, *im.size)
    w, h = im.size
    l, t = (w - s) // 2, (h - s) // 2
    return im.crop((l, t, l + s, t + s)).resize((256, 256), Image.BILINEAR)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="LungHist700 image root (contains aca_bd/, ...)")
    ap.add_argument("--csv", default="data_splits/lunghist700.csv")
    ap.add_argument("--folds", default="data_splits/folds_patient.csv")
    ap.add_argument("--out", default="results_rev1/WP01b")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    df = pd.read_csv(a.csv)
    fm = pd.read_csv(a.folds)
    df["fold"] = df.path.map(dict(zip(fm.path, fm.fold))).astype(int)
    y = np.array([lab(v) for v in df.label])
    groups = df.patient.to_numpy()

    rs = df.sample(300, random_state=SEED)
    st = [rgb_to_lab(np.asarray(load(os.path.join(a.root, p)), dtype=np.float32) / 255.0).reshape(-1, 3)
          for p in rs.path]
    target = (np.mean([s.mean(0) for s in st], 0), np.mean([s.std(0) for s in st], 0))

    cache = os.path.join(a.out, "wp01b_features.npz")
    if os.path.exists(cache):
        z = np.load(cache)
        F = {k: z[k] for k in z.files}
    else:
        F = {}
        for p in df.path:
            for k, v in features_for(load(os.path.join(a.root, p)), target).items():
                F.setdefault(k, []).append(v)
        F = {k: np.stack(v) for k, v in F.items()}
        np.savez_compressed(cache, **F)

    rows = []
    for fname, X in F.items():
        for cname in ("logreg", "random_forest"):
            pred = np.zeros(len(df), dtype=int)
            for f in range(5):
                te = df.fold.to_numpy() == f
                clf = classifiers()[cname]
                clf.fit(X[~te], y[~te])
                pred[te] = clf.predict(X[te])
            lo, hi = cluster_bootstrap(y, pred, groups)
            rec = recall_score(y, pred, average=None, labels=[0, 1, 2])
            rows.append(dict(features=fname, classifier=cname, n=len(df),
                             acc=round(100 * accuracy_score(y, pred), 2),
                             ci_low=round(lo, 1), ci_high=round(hi, 1),
                             macro_f1=round(f1_score(y, pred, average="macro"), 4),
                             recall_ACA=round(100 * rec[0], 1), recall_N=round(100 * rec[1], 1),
                             recall_SCC=round(100 * rec[2], 1)))
            print(f"{fname:17s} {cname:13s} acc = {rows[-1]['acc']:6.2f}% [{lo:.1f}-{hi:.1f}]")

    R = pd.DataFrame(rows)
    R.to_csv(os.path.join(a.out, "wp01b_results.csv"), index=False)
    with open(os.path.join(a.out, "wp01b_results.md"), "w") as fh:
        fh.write("| Features | Classifier | OOF acc. (%) [95% cluster CI] | Macro F1 | Recall ACA / N / SCC (%) |\n")
        fh.write("| --- | --- | --- | --- | --- |\n")
        for r in rows:
            fh.write(f"| {r['features']} | {r['classifier']} | {r['acc']:.2f} [{r['ci_low']}-{r['ci_high']}] | "
                     f"{r['macro_f1']:.4f} | {r['recall_ACA']} / {r['recall_N']} / {r['recall_SCC']} |\n")
    ch = R[(R.features == "colour_hist")].set_index("classifier").acc
    print(f"\nFill in the manuscript: colour-histogram RF = {ch['random_forest']:.2f}%, LR = {ch['logreg']:.2f}%")


if __name__ == "__main__":
    main()
