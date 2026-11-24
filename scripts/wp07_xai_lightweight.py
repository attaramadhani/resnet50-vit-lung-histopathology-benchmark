"""wp07_xai_lightweight.py - Lightweight WP07 XAI revision package for CMPB Update.

Implements the three XAI items requested for supplement / Reviewer 5 responses:
  * WP07-c (5.9c): Adebayo cascading parameter randomization (top-to-bottom)
                   measuring Spearman rank correlation and SSIM vs original map.
  * WP07-a (5.9a): Spatially-matched Gaussian-blurred random baseline calibrated to
                   match the spatial blob scale (FWHM) of Grad-CAM / Rollout, then
                   evaluating deletion and insertion AUC (K=10, K=20, grey baseline).
  * WP07-d (5.9d): Per-class Median and IQR across all methods (Method, Independent
                   Random, Spatially-Matched Random) on the 150-image cohort.
"""
import argparse
import copy
import json
import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from scipy.stats import spearmanr, wilcoxon

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

from common import build_model, load_checkpoint, resolve_checkpoint_path, eval_transform, ImageList, MEAN, STD  # noqa: E402
from exp2_faithfulness import gradcam, rollout, baseline_image, eval_curve, trapezoidal_auc  # noqa: E402


def ssim_2d(a: np.ndarray, b: np.ndarray, ksize: int = 7) -> float:
    """Compute SSIM between two [0, 1]-normalized 2D saliency maps (uniform ksize window)."""
    c1 = 0.01 ** 2
    c2 = 0.03 ** 2
    ta = torch.from_numpy(a.astype(np.float32))[None, None]
    tb = torch.from_numpy(b.astype(np.float32))[None, None]
    pad = ksize // 2
    mu_a = F.avg_pool2d(ta, ksize, stride=1, padding=pad)
    mu_b = F.avg_pool2d(tb, ksize, stride=1, padding=pad)
    mu_a2, mu_b2, mu_ab = mu_a * mu_a, mu_b * mu_b, mu_a * mu_b
    sig_a2 = F.avg_pool2d(ta * ta, ksize, stride=1, padding=pad) - mu_a2
    sig_b2 = F.avg_pool2d(tb * tb, ksize, stride=1, padding=pad) - mu_b2
    sig_ab = F.avg_pool2d(ta * tb, ksize, stride=1, padding=pad) - mu_ab
    num = (2.0 * mu_ab + c1) * (2.0 * sig_ab + c2)
    den = (mu_a2 + mu_b2 + c1) * (sig_a2 + sig_b2 + c2)
    return float((num / (den + 1e-12)).mean().item())


def normalize_map(t: torch.Tensor) -> np.ndarray:
    arr = t.detach().cpu().numpy().astype(np.float64)
    mn, mx = float(arr.min()), float(arr.max())
    if mx - mn < 1e-12:
        return np.zeros_like(arr)
    return (arr - mn) / (mx - mn)


def estimate_fwhm_pixels(maps_norm: list) -> float:
    """Estimate mean isotropic autocorrelation half-max width (pixels) across saliency maps."""
    radii = []
    for m in maps_norm:
        m0 = m - m.mean()
        var = float((m0 ** 2).sum())
        if var < 1e-12:
            continue
        f = np.fft.rfft2(m0)
        ac = np.fft.irfft2(f * np.conj(f), s=m0.shape)
        ac = np.fft.fftshift(ac)
        ac /= (ac.max() + 1e-12)
        cy, cx = np.array(ac.shape) // 2
        row = ac[cy, cx:]
        col = ac[cy:, cx]
        r_x = int(np.argmax(row < 0.5)) if np.any(row < 0.5) else len(row) // 2
        r_y = int(np.argmax(col < 0.5)) if np.any(col < 0.5) else len(col) // 2
        radii.append(0.5 * (r_x + r_y))
    return float(np.mean(radii)) if radii else 16.0


def reinit_module(mod: nn.Module, gen: torch.Generator):
    for m in mod.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            std = float(m.weight.detach().std().item()) if m.weight.numel() > 1 else 0.02
            std = max(std, 1e-3)
            with torch.no_grad():
                w = torch.empty_like(m.weight, device="cpu").normal_(0.0, std, generator=gen)
                m.weight.copy_(w.to(m.weight.device))
                if m.bias is not None:
                    m.bias.zero_()
        elif isinstance(m, (nn.BatchNorm2d, nn.LayerNorm)):
            with torch.no_grad():
                if m.weight is not None:
                    m.weight.fill_(1.0)
                if m.bias is not None:
                    m.bias.zero_()


def get_cascading_stages(arch: str):
    if arch == "resnet50":
        return [
            ("fc", lambda m: [m.fc]),
            ("layer4", lambda m: [m.layer4]),
            ("layer3", lambda m: [m.layer3]),
            ("layer2", lambda m: [m.layer2]),
            ("layer1", lambda m: [m.layer1]),
            ("conv1_bn1", lambda m: [m.conv1, m.bn1]),
        ]
    return [
        ("head", lambda m: [m.head]),
        ("norm", lambda m: [m.norm]),
        ("blocks_11_9", lambda m: [m.blocks[i] for i in [11, 10, 9]]),
        ("blocks_8_6", lambda m: [m.blocks[i] for i in [8, 7, 6]]),
        ("blocks_5_3", lambda m: [m.blocks[i] for i in [5, 4, 3]]),
        ("blocks_2_0", lambda m: [m.blocks[i] for i in [2, 1, 0]]),
        ("patch_embed", lambda m: [m.patch_embed]),
    ]


def gaussian_blur_2d(sal_2d: torch.Tensor, sigma: float) -> torch.Tensor:
    k = int(max(3, round(sigma * 6) | 1))
    k = min(k, 223)
    out = TF.gaussian_blur(sal_2d.unsqueeze(0).unsqueeze(0), kernel_size=k, sigma=sigma)
    return out[0, 0]


def run_arch(arch: str, ckpt_path: str, df: pd.DataFrame, root_dir: str, out_dir: str, seed: int = 42):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    resolved_ckpt = resolve_checkpoint_path(ckpt_path)
    base_model = load_checkpoint(build_model(arch, num_classes=3, pretrained=False), resolved_ckpt, dev).to(dev).eval()
    ds = ImageList(df, eval_transform("resize"), root_dir)
    mean = torch.tensor(MEAN, device=dev).view(1, 3, 1, 1)
    std = torch.tensor(STD, device=dev).view(1, 3, 1, 1)

    # 1. Precompute original saliency maps & target class
    orig_maps_tensor, orig_maps_norm, orig_preds = [], [], []
    xs_cache = []
    for i in range(len(ds)):
        x, y, _ = ds[i]
        x = x.unsqueeze(0).to(dev)
        with torch.no_grad():
            c = int(base_model(x).argmax())
        sal = gradcam(base_model, x.clone().requires_grad_(True), c) if arch == "resnet50" else rollout(base_model, x.clone(), dev)
        xs_cache.append((x, y, c))
        orig_maps_tensor.append(sal.detach())
        orig_maps_norm.append(normalize_map(sal))
        orig_preds.append(c)

    # 2. Calibrate spatially-matched Gaussian sigma from mean FWHM
    mean_r_half = estimate_fwhm_pixels(orig_maps_norm)
    # 2D Gaussian ACF half-max radius R relates to smoothing sigma of white noise via sigma_filter ≈ R / sqrt(2 * ln(2))
    sigma_match = float(np.clip(mean_r_half / 1.1774, 4.0, 45.0))
    print(f"[{arch}] Mean saliency ACF half-max radius = {mean_r_half:.2f} px -> calibrated Gaussian random sigma = {sigma_match:.2f}")

    # 3. WP07-a: Faithfulness on Method vs Independent Random vs Spatially-Matched Random (grey baseline, K=10 & K=20)
    rec_a = []
    for i, (x, y, c) in enumerate(xs_cache):
        sal_m = orig_maps_tensor[i]
        gen = torch.Generator(device="cpu").manual_seed(seed + i * 1000 + (1 if arch == "resnet50" else 2))
        sal_rand_indep = torch.rand(x.shape[-2:], generator=gen).to(dev)
        sal_rand_spatial = gaussian_blur_2d(sal_rand_indep, sigma_match)

        x_raw = x * std + mean
        base_img = baseline_image(x_raw, "grey")
        for mname, sal in [("method", sal_m), ("random_indep", sal_rand_indep), ("random_spatial_matched", sal_rand_spatial)]:
            row = {
                "arch": arch,
                "idx": i,
                "image_path": df.iloc[i]["path"],
                "true_label": int(y),
                "pred_target": int(c),
                "method": mname,
                "sigma_match": round(sigma_match, 3),
            }
            for K_val in [10, 20]:
                for mode in ["deletion", "insertion"]:
                    probs, logits, _ = eval_curve(base_model, x, c, sal, base_img, mode, K_val, mean, std, dev)
                    row[f"{mode}_auc_prob_k{K_val}"] = trapezoidal_auc(probs)
                    row[f"{mode}_auc_logit_k{K_val}"] = trapezoidal_auc(logits)
            rec_a.append(row)

    df_a = pd.DataFrame(rec_a)
    df_a.to_csv(os.path.join(out_dir, f"wp07a_{arch}_per_image.csv"), index=False)

    # 4. WP07-c: Adebayo cascading parameter randomization (top-to-bottom)
    rec_c = []
    casc_model = copy.deepcopy(base_model).to(dev).eval()
    rgen = torch.Generator(device="cpu").manual_seed(seed + 777)
    for stage_idx, (stage_name, getter) in enumerate(get_cascading_stages(arch), 1):
        for mod in getter(casc_model):
            reinit_module(mod, rgen)
        for i, (x, y, c) in enumerate(xs_cache):
            sal_rand = gradcam(casc_model, x.clone().requires_grad_(True), c) if arch == "resnet50" else rollout(casc_model, x.clone(), dev)
            norm_rand = normalize_map(sal_rand)
            norm_orig = orig_maps_norm[i]
            rho, _ = spearmanr(norm_orig.ravel(), norm_rand.ravel())
            if np.isnan(rho):
                rho = 0.0
            ssim_val = ssim_2d(norm_orig, norm_rand, ksize=7)
            rec_c.append({
                "arch": arch,
                "stage_order": stage_idx,
                "stage": stage_name,
                "idx": i,
                "true_label": int(y),
                "spearman_rho": float(rho),
                "ssim": float(ssim_val),
            })

    df_c = pd.DataFrame(rec_c)
    df_c.to_csv(os.path.join(out_dir, f"wp07c_{arch}_adebayo_per_image.csv"), index=False)
    print(f"[OK] Finished WP07-a & WP07-c for {arch}")
    return df_a, df_c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="LC25000 root directory")
    ap.add_argument("--images-cache", default="data_splits/exp2_images.csv")
    ap.add_argument("--r50-ckpt", required=True)
    ap.add_argument("--vit-ckpt", required=True)
    ap.add_argument("--out", default="results_rev1/WP07")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    df = pd.read_csv(args.images_cache)

    df_a_r50, df_c_r50 = run_arch("resnet50", args.r50_ckpt, df, args.root, args.out, args.seed)
    df_a_vit, df_c_vit = run_arch("vit_s16", args.vit_ckpt, df, args.root, args.out, args.seed)

    all_a = pd.concat([df_a_r50, df_a_vit], ignore_index=True)
    all_c = pd.concat([df_c_r50, df_c_vit], ignore_index=True)

    # WP07-a summary + Wilcoxon tests
    a_summ = []
    for arch in ["resnet50", "vit_s16"]:
        sub_arch = all_a[all_a["arch"] == arch]
        m_df = sub_arch[sub_arch["method"] == "method"].sort_values("idx")
        for ref in ["random_indep", "random_spatial_matched"]:
            r_df = sub_arch[sub_arch["method"] == ref].sort_values("idx")
            for col in ["deletion_auc_prob_k10", "insertion_auc_prob_k10", "deletion_auc_prob_k20", "insertion_auc_prob_k20"]:
                d = m_df[col].to_numpy() - r_df[col].to_numpy()
                nz = d[d != 0]
                pval = float(wilcoxon(d).pvalue) if len(nz) > 0 else 1.0
                a_summ.append({
                    "arch": arch,
                    "metric": col,
                    "baseline_type": ref,
                    "method_mean": round(float(m_df[col].mean()), 4),
                    "method_std": round(float(m_df[col].std()), 4),
                    "baseline_mean": round(float(r_df[col].mean()), 4),
                    "baseline_std": round(float(r_df[col].std()), 4),
                    "mean_diff": round(float(d.mean()), 4),
                    "wilcoxon_p": pval,
                })
    pd.DataFrame(a_summ).to_csv(os.path.join(args.out, "wp07a_spatial_baseline_summary.csv"), index=False)

    # WP07-c summary
    c_rows = []
    for (arch, st_ord, stage), g in all_c.groupby(["arch", "stage_order", "stage"]):
        c_rows.append({
            "arch": arch,
            "stage_order": int(st_ord),
            "stage": stage,
            "spearman_mean": round(float(g["spearman_rho"].mean()), 4),
            "spearman_std": round(float(g["spearman_rho"].std()), 4),
            "spearman_median": round(float(g["spearman_rho"].median()), 4),
            "spearman_iqr": round(float(g["spearman_rho"].quantile(0.75) - g["spearman_rho"].quantile(0.25)), 4),
            "ssim_mean": round(float(g["ssim"].mean()), 4),
            "ssim_std": round(float(g["ssim"].std()), 4),
            "ssim_median": round(float(g["ssim"].median()), 4),
            "ssim_iqr": round(float(g["ssim"].quantile(0.75) - g["ssim"].quantile(0.25)), 4),
        })
    pd.DataFrame(c_rows).to_csv(os.path.join(args.out, "wp07c_adebayo_summary.csv"), index=False)

    # WP07-d Median and IQR per class
    cls_names = {0: "ACA", 1: "N", 2: "SCC"}
    d_rows = []
    for (arch, mname, cls_id), g in all_a.groupby(["arch", "method", "true_label"]):
        for col in ["deletion_auc_prob_k10", "insertion_auc_prob_k10", "deletion_auc_prob_k20", "insertion_auc_prob_k20"]:
            q25, med, q75 = g[col].quantile([0.25, 0.50, 0.75])
            d_rows.append({
                "arch": arch,
                "method": mname,
                "class_id": int(cls_id),
                "class_name": cls_names.get(int(cls_id), str(cls_id)),
                "metric": col,
                "n": len(g),
                "median": round(float(med), 4),
                "q25": round(float(q25), 4),
                "q75": round(float(q75), 4),
                "iqr": round(float(q75 - q25), 4),
            })
    pd.DataFrame(d_rows).to_csv(os.path.join(args.out, "wp07d_median_iqr_per_class.csv"), index=False)

    with open(os.path.join(args.out, "wp07_summary.json"), "w", encoding="utf-8") as f:
        json.dump({"wp07a": a_summ, "wp07c": c_rows, "wp07d": d_rows}, f, indent=2)
    print(f"[OK] Saved complete WP07 lightweight outputs to {args.out}")


if __name__ == "__main__":
    main()
