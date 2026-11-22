"""Exp 2 - Faithfulness re-test (manuscript Section 3.6 / Table 5 / 5.5).

Implements Section 6 of implementation_plan2.md:
  * Model: checkpoint fold0 for scenario 1 (ResNet50) and scenario 6 (ViT-S/16), SHA256 recorded.
  * Images: 150 test images (50 per class), seed 42; saved/loaded from exp2_images.csv.
  * Attribution methods:
      - ResNet50: Grad-CAM on layer4 from raw logit
      - ViT-S/16: Attention rollout with residual correction across all 12 blocks
  * Main analysis (manuscript Table 5):
      - K = 10 (10% pixels per step)
      - Median grey baseline (canvas insertion / grey deletion)
      - Softmax probability curves and trapezoidal AUC (Eq. 1)
  * Sensitivity analysis:
      - K = 20 (5% pixels per step)
      - Blur (Gaussian kernel=31, sigma=10) and tissue-mean baselines
      - Raw logit curves and AUC
  * Random attribution baseline computed SEPARATELY for each model
  * Statistical validation: Wilcoxon signed-rank test (method vs random baseline per image)
  * Collapse diagnostic: predicted class after step 1
"""
import argparse
import hashlib
import json
import os
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.stats import wilcoxon

from common import build_model, load_checkpoint, resolve_checkpoint_path, eval_transform, ImageList, MEAN, STD


def sha256_file(filepath):
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------- attribution methods
def gradcam(model, x, c):
    feats = {}
    layer = model.layer4
    h1 = layer.register_forward_hook(lambda m, i, o: feats.__setitem__("a", o))
    h2 = layer.register_full_backward_hook(lambda m, gi, go: feats.__setitem__("g", go[0]))
    model.zero_grad()
    logit = model(x)[0, c]
    logit.backward()
    h1.remove()
    h2.remove()
    w = feats["g"].mean((2, 3), keepdim=True)
    cam = F.relu((w * feats["a"]).sum(1, keepdim=True))
    return F.interpolate(cam, x.shape[-2:], mode="bilinear", align_corners=False)[0, 0].detach()


def rollout(model, x, dev):
    """Attention rollout with residual correction (Abnar & Zuidema) for timm ViT."""
    atts = []

    def hook(mod, inp, out):
        xx = inp[0]
        B, N, C = xx.shape
        qkv = mod.qkv(xx).reshape(B, N, 3, mod.num_heads, C // mod.num_heads).permute(2, 0, 3, 1, 4)
        q, k = qkv[0], qkv[1]
        q, k = mod.q_norm(q), mod.k_norm(k)
        atts.append(((q @ k.transpose(-2, -1)) * mod.scale).softmax(-1).mean(1).detach())

    hs = [b.attn.register_forward_hook(hook) for b in model.blocks]
    with torch.no_grad():
        model(x)
    for h in hs:
        h.remove()

    R = torch.eye(atts[0].shape[-1], device=dev)
    for A in atts:
        A = A[0] + torch.eye(A.shape[-1], device=dev)
        A = A / A.sum(-1, keepdim=True)
        R = A @ R

    n_prefix = getattr(model, "num_prefix_tokens", 1)
    m = R[0, n_prefix:]
    s = int(m.numel() ** 0.5)
    return F.interpolate(m.reshape(1, 1, s, s), x.shape[-2:], mode="bilinear", align_corners=False)[0, 0]


def baseline_image(x_raw, kind):
    if kind == "grey":
        return torch.full_like(x_raw, float(x_raw.median()))
    if kind == "blur":
        import torchvision.transforms.functional as TF
        return TF.gaussian_blur(x_raw, kernel_size=31, sigma=10.0)
    if kind == "tissue_mean":
        return x_raw.mean((2, 3), keepdim=True).expand_as(x_raw)
    raise ValueError(f"Unknown baseline kind: {kind}")


@torch.no_grad()
def eval_curve(model, x, c, sal, base_img, mode, K, mean, std, dev):
    """Computes perturbation curve for a specific K and mode."""
    x_raw = x * std + mean
    order = torch.argsort(sal.flatten(), descending=True)
    HW = order.numel()
    step = HW // K
    mask = torch.zeros(HW, device=dev)
    imgs = []

    for k in range(K + 1):
        if k > 0:
            end_idx = k * step if k < K else HW
            mask[order[(k - 1) * step: end_idx]] = 1.0
        m = mask.view(1, 1, *x.shape[-2:])
        if mode == "deletion":
            # Deletion: replace salient pixels with baseline
            img = x_raw * (1.0 - m) + base_img * m
        else:
            # Insertion: insert salient pixels into baseline canvas
            img = base_img * (1.0 - m) + x_raw * m
        imgs.append((img - mean) / std)

    batch = torch.cat(imgs)
    out = model(batch).float()
    probs = out.softmax(1)[:, c].cpu().numpy()
    logits = out[:, c].cpu().numpy()
    pred_step1 = int(out[1].argmax())
    return probs, logits, pred_step1


def trapezoidal_auc(v):
    """Eq. (1) in Section 3.6: AUC under perturbation curve."""
    return float(np.mean([(v[k - 1] + v[k]) / 2.0 for k in range(1, len(v))]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", required=True, choices=["resnet50", "vit_s16", "vit_s"])
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--csv", default="lc25000_test.csv")
    ap.add_argument("--root", default="")
    ap.add_argument("--n-per-class", type=int, default=50)
    ap.add_argument("--images-cache", default="exp2_images.csv")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    arch_name = "vit_s16" if args.arch in ["vit_s16", "vit_s"] else "resnet50"
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print(f"\n==========================================================================")
    print(f"  EXP 2: FAITHFULNESS XAI BENCHMARK (Table 5 & Section 3.6)")
    print(f"  Arch: {arch_name} | Device: {dev}")
    resolved_ckpt = resolve_checkpoint_path(args.ckpt)
    print(f"  Checkpoint: {resolved_ckpt}")
    if os.path.exists(resolved_ckpt):
        ckpt_hash = sha256_file(resolved_ckpt)
        print(f"  Checkpoint SHA256: {ckpt_hash}")
        args.ckpt = resolved_ckpt
    else:
        raise FileNotFoundError(f"Checkpoint not found: {args.ckpt} (or resolved aliases)")
    print(f"==========================================================================")

    # 1. Manage the 150 test image cohort (seed 42, 50 per class)
    if os.path.exists(args.images_cache):
        print(f"Loading existing 150-image subset from: {args.images_cache}")
        df = pd.read_csv(args.images_cache)
    else:
        print(f"Sampling 150 images (50/class) from {args.csv} with seed {args.seed}...")
        raw_df = pd.read_csv(args.csv)
        df = raw_df.sample(frac=1.0, random_state=args.seed).groupby("label").head(args.n_per_class).reset_index(drop=True)
        df.to_csv(args.images_cache, index=False)
        print(f"Saved cohort to {args.images_cache} ({len(df)} images: {df['label'].value_counts().to_dict()})")

    ds = ImageList(df, eval_transform("resize"), args.root)
    mean = torch.tensor(MEAN, device=dev).view(1, 3, 1, 1)
    std = torch.tensor(STD, device=dev).view(1, 3, 1, 1)

    # Build and load model
    model = build_model(arch_name, num_classes=3, pretrained=False)
    model = load_checkpoint(model, args.ckpt, dev).to(dev).eval()

    records = []
    print(f"Starting faithfulness evaluation over {len(ds)} images...")

    for i in range(len(ds)):
        x, y, _ = ds[i]
        x = x.unsqueeze(0).to(dev)

        # Target class = model's initial predicted class
        with torch.no_grad():
            c = int(model(x).argmax())

        # 1. Method saliency
        if arch_name == "resnet50":
            sal_m = gradcam(model, x.clone().requires_grad_(True), c)
        else:
            sal_m = rollout(model, x.clone(), dev)

        # 2. Random baseline (independent RNG per image and per model)
        gen = torch.Generator(device="cpu").manual_seed(args.seed + i * 1000 + (1 if arch_name == "resnet50" else 2))
        sal_r = torch.rand(x.shape[-2:], generator=gen).to(dev)

        x_raw = x * std + mean
        baselines = {
            "grey": baseline_image(x_raw, "grey"),
            "blur": baseline_image(x_raw, "blur"),
            "tissue_mean": baseline_image(x_raw, "tissue_mean")
        }

        for method_type, sal in [("method", sal_m), ("random", sal_r)]:
            for base_name, base_img in baselines.items():
                rec = {
                    "idx": i,
                    "image_path": df.iloc[i]["path"],
                    "true_label": y,
                    "pred_target": c,
                    "method": method_type,
                    "baseline": base_name
                }
                for K_val in [10, 20]:
                    for mode in ["deletion", "insertion"]:
                        probs, logits, p_step1 = eval_curve(model, x, c, sal, base_img, mode, K_val, mean, std, dev)
                        rec[f"{mode}_auc_prob_k{K_val}"] = trapezoidal_auc(probs)
                        rec[f"{mode}_auc_logit_k{K_val}"] = trapezoidal_auc(logits)
                        rec[f"{mode}_pred_step1_k{K_val}"] = p_step1
                        rec[f"{mode}_p_start_k{K_val}"] = float(probs[0])
                        rec[f"{mode}_p_end_k{K_val}"] = float(probs[-1])
                records.append(rec)

        if (i + 1) % 25 == 0 or (i + 1) == len(ds):
            print(f"  Processed {i + 1}/{len(ds)} images...")

    R = pd.DataFrame(records)
    csv_out = args.out + "_per_image.csv"
    R.to_csv(csv_out, index=False)
    print(f"\n[OK] Per-image records saved to {csv_out}")

    # Summary statistics
    # Primary: K=10, median grey (Table 5 in draft)
    # Sensitivity: K=20, blur, tissue-mean, logit
    metrics_to_agg = [
        "deletion_auc_prob_k10", "insertion_auc_prob_k10",
        "deletion_auc_prob_k20", "insertion_auc_prob_k20",
        "deletion_auc_logit_k10", "insertion_auc_logit_k10",
        "deletion_auc_logit_k20", "insertion_auc_logit_k20"
    ]
    summ = R.groupby(["method", "baseline"])[metrics_to_agg].agg(["mean", "std"]).round(4)
    per_class = R.groupby(["method", "baseline", "true_label"])[["deletion_auc_prob_k10", "insertion_auc_prob_k10"]].mean().round(4)
    collapse_k10 = R[(R.baseline == "grey")].groupby("method")["deletion_pred_step1_k10"].value_counts(normalize=True).round(3)

    # Wilcoxon signed-rank tests: Method vs Random
    wilcoxon_tests = {}
    for base_name in ["grey", "blur", "tissue_mean"]:
        m_sub = R[(R.method == "method") & (R.baseline == base_name)].sort_values("idx")
        r_sub = R[(R.method == "random") & (R.baseline == base_name)].sort_values("idx")
        for col in metrics_to_agg:
            d = m_sub[col].to_numpy() - r_sub[col].to_numpy()
            diff_nz = d[d != 0]
            if len(diff_nz) > 0:
                stat, pval = wilcoxon(d)
                pval = float(pval)
            else:
                pval = 1.0
            wilcoxon_tests[f"{base_name}/{col}"] = {
                "mean_diff": round(float(np.mean(d)), 4),
                "p_value": pval,
                "significant_005": pval < 0.05
            }

    # Format summary text and json
    summary_txt = args.out + "_summary.txt"
    summary_json = args.out + "_summary.json"

    with open(summary_txt, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write(f"EXP 2 FAITHFULNESS SUMMARY: {arch_name.upper()}\n")
        f.write(f"Checkpoint: {args.ckpt}\n")
        f.write(f"SHA256: {ckpt_hash}\n")
        f.write("=" * 80 + "\n\n")
        f.write("== 1. OVERALL AUC (Mean +/- SD) ==\n")
        f.write(summ.to_string() + "\n\n")
        f.write("== 2. PER-CLASS PRIMARY AUC (K=10, Softmax Prob) ==\n")
        f.write(per_class.to_string() + "\n\n")
        f.write("== 3. COLLAPSE DIAGNOSTIC: PREDICTED CLASS AFTER STEP 1 (Grey Baseline, K=10) ==\n")
        f.write(collapse_k10.to_string() + "\n\n")
        f.write("== 4. METHOD vs RANDOM (Wilcoxon Signed-Rank Test, Paired by Image) ==\n")
        f.write(json.dumps(wilcoxon_tests, indent=2) + "\n")

    json_payload = {
        "arch": arch_name,
        "checkpoint": args.ckpt,
        "checkpoint_sha256": ckpt_hash,
        "wilcoxon_tests": wilcoxon_tests
    }
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(json_payload, f, indent=2)

    print(f"[OK] Summary saved to {summary_txt} and {summary_json}")
    # Print primary results
    print("\n--- PRIMARY RESULT (K=10, Median Grey Baseline) ---")
    grey_method = R[(R.method == "method") & (R.baseline == "grey")]
    grey_random = R[(R.method == "random") & (R.baseline == "grey")]
    print(f"Method Deletion AUC (prob) : {grey_method['deletion_auc_prob_k10'].mean():.4f} +/- {grey_method['deletion_auc_prob_k10'].std():.4f}")
    print(f"Random Deletion AUC (prob) : {grey_random['deletion_auc_prob_k10'].mean():.4f} +/- {grey_random['deletion_auc_prob_k10'].std():.4f}")
    print(f"Method Insertion AUC (prob): {grey_method['insertion_auc_prob_k10'].mean():.4f} +/- {grey_method['insertion_auc_prob_k10'].std():.4f}")
    print(f"Random Insertion AUC (prob): {grey_random['insertion_auc_prob_k10'].mean():.4f} +/- {grey_random['insertion_auc_prob_k10'].std():.4f}")


if __name__ == "__main__":
    main()
