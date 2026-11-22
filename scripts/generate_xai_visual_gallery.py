"""Generate the multi-case XAI visual gallery (Fig. 6) using v2.0.0 ResNet50 and ViT-S/16 models."""
import argparse
import os
import sys
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

from common import build_model, load_checkpoint, eval_transform, MEAN, STD  # noqa: E402
from exp2_faithfulness import gradcam, rollout  # noqa: E402


CASES = [
    ("lung_aca", "lungaca556.jpeg", "Case 1: Adenocarcinoma"),
    ("lung_aca", "lungaca678.jpeg", "Case 2: Adenocarcinoma"),
    ("lung_n", "lungn129.jpeg", "Case 3: Benign Tissue"),
    ("lung_n", "lungn64.jpeg", "Case 4: Benign Tissue"),
    ("lung_scc", "lungscc1015.jpeg", "Case 5: Squamous Cell Carcinoma"),
    ("lung_scc", "lungscc1314.jpeg", "Case 6: Squamous Cell Carcinoma"),
]
CLASSES = ["ACA", "N", "SCC"]


def overlay_turbo(rgb_float, sal_map, alpha=0.45):
    sal = sal_map.detach().cpu().numpy()
    sal = (sal - sal.min()) / (sal.max() - sal.min() + 1e-8)
    heat = cm.turbo(sal)[..., :3]
    return np.clip((1.0 - alpha) * rgb_float + alpha * heat, 0.0, 1.0), sal


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True, help="LC25000 root containing lung_aca, lung_n, lung_scc")
    ap.add_argument("--r50-ckpt", default="runs/resnet50_main/fold0.pt")
    ap.add_argument("--vit-ckpt", default="runs/vit_s_main/fold0.pt")
    ap.add_argument("--out", default="figures/fig6_xai_multiclass_gallery.png")
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    m_res = load_checkpoint(build_model("resnet50", pretrained=False), args.r50_ckpt, dev).to(dev).eval()
    m_vit = load_checkpoint(build_model("vit_s16", pretrained=False), args.vit_ckpt, dev).to(dev).eval()

    tf = eval_transform("resize")
    mean = np.array(MEAN).reshape(1, 1, 3)
    std = np.array(STD).reshape(1, 1, 3)

    fig, axes = plt.subplots(len(CASES), 4, figsize=(13, 18), dpi=300)
    col_titles = ["1. Original H&E Stain", "2. ResNet50 Grad-CAM", "3. ViT-S/16 Attention Rollout", "4. ViT-S/16 Saliency Map"]

    for row_idx, (cls_dir, fname, case_lbl) in enumerate(CASES):
        img_path = os.path.join(args.data_root, cls_dir, fname)
        im = Image.open(img_path).convert("RGB")
        x = tf(im).unsqueeze(0).to(dev)
        rgb = np.clip(x[0].permute(1, 2, 0).cpu().numpy() * std + mean, 0.0, 1.0)

        with torch.no_grad():
            p_res = torch.softmax(m_res(x), 1)[0]
            p_vit = torch.softmax(m_vit(x), 1)[0]
        c_res, c_vit = int(p_res.argmax()), int(p_vit.argmax())

        sal_res = gradcam(m_res, x.clone().requires_grad_(True), c_res)
        sal_vit = rollout(m_vit, x.clone(), dev)

        ov_res, _ = overlay_turbo(rgb, sal_res)
        ov_vit, sal_vit_norm = overlay_turbo(rgb, sal_vit)

        panels = [rgb, ov_res, ov_vit, sal_vit_norm]
        subtitles = [
            f"{case_lbl}\n[{fname}]",
            f"Pred: {CLASSES[c_res]} ({p_res[c_res]*100:.1f}%)",
            f"Pred: {CLASSES[c_vit]} ({p_vit[c_vit]*100:.1f}%)",
            "ViT-S Attention Flow",
        ]
        for col_idx, (panel, st) in enumerate(zip(panels, subtitles)):
            ax = axes[row_idx, col_idx]
            if col_idx == 3:
                ax.imshow(panel, cmap="turbo")
            else:
                ax.imshow(panel)
            if row_idx == 0:
                ax.set_title(f"{col_titles[col_idx]}\n{st}", fontsize=9, fontweight="bold")
            else:
                ax.set_title(st, fontsize=8.5)
            ax.set_xticks([])
            ax.set_yticks([])

    plt.tight_layout()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    plt.savefig(args.out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[OK] Saved XAI gallery to {args.out}")


if __name__ == "__main__":
    main()
