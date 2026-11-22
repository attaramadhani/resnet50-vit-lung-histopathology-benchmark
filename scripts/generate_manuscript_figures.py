"""Reproduce publication figures (Figs 2-5, 7-8, and Fig S1) directly from logs_and_metrics/."""
import json
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

LOGS = "logs_and_metrics"
FIGS = "figures"
os.makedirs(FIGS, exist_ok=True)

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial"],
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 300,
})

BLUE = "#1D65A6"
ORANGE = "#D95F02"
GREY = "#94A3B8"


def load_run(sc):
    with open(os.path.join(LOGS, "runs", sc, "metrics.json"), "r", encoding="utf-8") as f:
        return json.load(f)


def fig2_scenarios():
    r50_list = [
        ("resnet50_main", "ResNet50 · Adam 1e-4"),
        ("resnet50_rmsprop", "ResNet50 · RMSprop 1e-4"),
        ("resnet50_lr_1e3", "ResNet50 · Adam 1e-3"),
        ("resnet50_lr_1e5", "ResNet50 · Adam 1e-5"),
        ("resnet50_sgd", "ResNet50 · SGD 1e-4"),
        ("resnet50_sgdm", "ResNet50 · SGD+m 1e-4"),
        ("resnet50_sgdm_lr_1e3", "ResNet50 · SGD+m 1e-3"),
    ]
    vit_list = [
        ("vit_s_main", "ViT-S/16 · AdamW 1e-4"),
        ("vit_s_rmsprop", "ViT-S/16 · RMSprop 1e-4"),
        ("vit_s_lr_1e3", "ViT-S/16 · AdamW 1e-3"),
        ("vit_s_lr_1e5", "ViT-S/16 · AdamW 1e-5"),
        ("vit_s_sgd", "ViT-S/16 · SGD 1e-4"),
        ("vit_s_sgdm", "ViT-S/16 · SGD+m 1e-4"),
        ("vit_s_sgdm_lr_1e3", "ViT-S/16 · SGD+m 1e-3"),
    ]
    all_items = [(sc, lbl, BLUE) for sc, lbl in r50_list] + [(sc, lbl, ORANGE) for sc, lbl in vit_list]
    n = len(all_items)
    y_pos = np.arange(n)[::-1]

    fig, ax = plt.subplots(figsize=(10.5, 5.6))
    labels = []
    for idx, (sc, lbl, col) in enumerate(all_items):
        d = load_run(sc)
        cv_m = d["cv_val_accuracy"]["mean"]
        cv_s = d["cv_val_accuracy"]["std"]
        te_a = d["ensemble_test"]["acc"]
        y = y_pos[idx]
        ax.errorbar(cv_m, y, xerr=cv_s, fmt="o", mfc="white", mec=col, ecolor=col, capsize=3, ms=6.5, lw=1.4, zorder=3)
        ax.scatter(te_a, y, marker="D", color=col, s=42, zorder=4)
        ax.text(100.95, y, f"{te_a:.2f}%", va="center", ha="left", fontsize=9, color=col, fontweight="bold")
        labels.append(lbl)

    ax.axhline(6.5, color="#CBD5E1", lw=1.0)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.set_xlim(93.0, 100.8)
    ax.set_xlabel("Accuracy (%)")
    ax.grid(axis="x", linestyle=":", alpha=0.5)
    ax.text(100.95, n - 0.1, "Test", fontweight="bold", fontsize=9.5, ha="left")
    ax.errorbar([], [], xerr=0, fmt="o", mfc="white", mec="#334155", ecolor="#334155", label="5-fold CV accuracy (mean ± SD, n = 2,400 per fold)")
    ax.scatter([], [], marker="D", color="#334155", label="Test accuracy, 5-fold ensemble (n = 3,000)")
    ax.legend(loc="lower left", frameon=True, fontsize=8.5)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGS, "fig2_lc25000_scenarios.png"), bbox_inches="tight")
    plt.close()


def fig3_sensitivity():
    main_r = load_run("resnet50_main")["ensemble_test"]["acc"]
    main_v = load_run("vit_s_main")["ensemble_test"]["acc"]
    pairs = [
        ("RMSprop\n1e-4", "resnet50_rmsprop", "vit_s_rmsprop"),
        ("Adam(W)\n1e-3", "resnet50_lr_1e3", "vit_s_lr_1e3"),
        ("Adam(W)\n1e-5", "resnet50_lr_1e5", "vit_s_lr_1e5"),
        ("SGD\n1e-4", "resnet50_sgd", "vit_s_sgd"),
        ("SGD+m\n1e-4", "resnet50_sgdm", "vit_s_sgdm"),
        ("SGD+m\n1e-3", "resnet50_sgdm_lr_1e3", "vit_s_sgdm_lr_1e3"),
    ]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.2))
    x = np.arange(len(pairs))
    w = 0.36
    dr = [load_run(r)["ensemble_test"]["acc"] - main_r for _, r, _ in pairs]
    dv = [load_run(v)["ensemble_test"]["acc"] - main_v for _, _, v in pairs]
    ax1.bar(x - w / 2, dr, w, color=BLUE, label="ResNet50")
    ax1.bar(x + w / 2, dv, w, color=ORANGE, label="ViT-S/16")
    ax1.axhline(0, color="black", lw=1.0)
    for i, (r_val, v_val) in enumerate(zip(dr, dv)):
        ax1.text(i - w / 2, r_val + (0.12 if r_val >= 0 else -0.35), f"{r_val:.2f}", ha="center", fontsize=8)
        ax1.text(i + w / 2, v_val + (0.12 if v_val >= 0 else -0.35), f"{v_val:.2f}", ha="center", fontsize=8)
    ax1.set_xticks(x)
    ax1.set_xticklabels([p[0] for p in pairs], fontsize=8.8)
    ax1.set_ylim(-5.8, 0.8)
    ax1.set_ylabel("Change in test accuracy vs main (percentage points)")
    ax1.set_title("a", loc="left", fontweight="bold")
    ax1.legend(frameon=False, loc="lower left")

    lrs = [1e-5, 1e-4, 1e-3]
    r_scs = ["resnet50_lr_1e5", "resnet50_main", "resnet50_lr_1e3"]
    v_scs = ["vit_s_lr_1e5", "vit_s_main", "vit_s_lr_1e3"]
    r_m = [load_run(s)["cv_val_accuracy"]["mean"] for s in r_scs]
    r_s = [load_run(s)["cv_val_accuracy"]["std"] for s in r_scs]
    v_m = [load_run(s)["cv_val_accuracy"]["mean"] for s in v_scs]
    v_s = [load_run(s)["cv_val_accuracy"]["std"] for s in v_scs]
    ax2.errorbar(lrs, r_m, yerr=r_s, fmt="-o", color=BLUE, capsize=4, label="ResNet50 (Adam)")
    ax2.errorbar(lrs, v_m, yerr=v_s, fmt="-s", color=ORANGE, capsize=4, label="ViT-S/16 (AdamW)")
    ax2.set_xscale("log")
    ax2.set_ylim(92.0, 100.6)
    ax2.set_xlabel("Learning rate")
    ax2.set_ylabel("5-fold CV accuracy (%, mean ± SD)")
    ax2.set_title("b", loc="left", fontweight="bold")
    ax2.legend(frameon=False, loc="lower left")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGS, "fig3_optimizer_sensitivity.png"), bbox_inches="tight")
    plt.close()


def fig4_compute():
    r = json.load(open(os.path.join(LOGS, "exp3_resnet50.json"), encoding="utf-8"))
    v = json.load(open(os.path.join(LOGS, "exp3_vit_s16.json"), encoding="utf-8"))
    panels = [
        ("a", "Training throughput\n(images/s)", [r["train_img_per_s"][0], v["train_img_per_s"][0]], [r["train_img_per_s"][1], v["train_img_per_s"][1]], "{:.0f}", 2100),
        ("b", "Inference latency per image\n(ms, batch 32)", [r["latency_ms_per_image_bs32"], v["latency_ms_per_image_bs32"]], [0, 0], "{:.3f}", 0.23),
        ("c", "Peak GPU memory\nduring training (GB)", [r["peak_mem_GB_train"], v["peak_mem_GB_train"]], [0, 0], "{:.2f}", 2.3),
        ("d", "MACs per 224 × 224\nimage (G)", [r["MACs_G"], v["MACs_G"]], [0, 0], "{:.2f}", 5.3),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(12.0, 3.2))
    for ax, (tag, ttl, vals, errs, fmt, ymax) in zip(axes, panels):
        bars = ax.bar(["ResNet50", "ViT-S/16"], vals, yerr=errs if any(errs) else None, capsize=4, color=[BLUE, ORANGE], width=0.55)
        for b, val in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, val + ymax * 0.04, fmt.format(val), ha="center", fontsize=9)
        ax.set_ylim(0, ymax)
        ax.set_title(ttl, fontsize=9.5)
        ax.text(-0.18, 1.05, tag, transform=ax.transAxes, fontweight="bold", fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGS, "fig4_compute_profile.png"), bbox_inches="tight")
    plt.close()


def fig5_faithfulness():
    dr = pd.read_csv(os.path.join(LOGS, "exp2_faithfulness", "exp2_resnet50_per_image.csv"))
    dv = pd.read_csv(os.path.join(LOGS, "exp2_faithfulness", "exp2_vit_per_image.csv"))
    baselines = [("a", "Median grey (primary)", "grey"), ("b", "Gaussian blur", "blur"), ("c", "Tissue mean colour", "tissue_mean")]
    fig, axes = plt.subplots(1, 3, figsize=(12.2, 4.0), sharey=True)
    x = np.arange(4)
    w = 0.36
    x_labels = ["Grad-CAM\n(ResNet50)", "Random\n(ResNet50)", "Rollout\n(ViT-S/16)", "Random\n(ViT-S/16)"]
    cols = [BLUE, GREY, ORANGE, GREY]
    for ax, (tag, ttl, bkey) in zip(axes, baselines):
        dels, del_sd, ins, ins_sd = [], [], [], []
        for df_m, m_type in [(dr, "method"), (dr, "random"), (dv, "method"), (dv, "random")]:
            sub = df_m[(df_m["baseline"] == bkey) & (df_m["method"] == m_type)]
            dels.append(sub["deletion_auc_prob_k10"].mean())
            del_sd.append(sub["deletion_auc_prob_k10"].std())
            ins.append(sub["insertion_auc_prob_k10"].mean())
            ins_sd.append(sub["insertion_auc_prob_k10"].std())
        for i in range(4):
            ax.bar(x[i] - w / 2, dels[i], w, yerr=del_sd[i], color=cols[i], alpha=0.6, hatch="//", edgecolor="#334155", capsize=3,
                   label="Deletion (lower = more faithful)" if (tag == "c" and i == 1) else None)
            ax.bar(x[i] + w / 2, ins[i], w, yerr=ins_sd[i], color=cols[i], alpha=0.95, edgecolor="#334155", capsize=3,
                   label="Insertion (higher = more faithful)" if (tag == "c" and i == 1) else None)
        ax.axvline(1.5, color="#E2E8F0", lw=1.2)
        ax.set_xticks(x)
        ax.set_xticklabels(x_labels, fontsize=8.2)
        ax.set_ylim(0, 1.18)
        ax.set_title(ttl, fontsize=10)
        ax.text(-0.08, 1.03, tag, transform=ax.transAxes, fontweight="bold", fontsize=11)
        if tag == "a":
            ax.set_ylabel("AUC, K = 10 (mean ± SD, n = 150)")
        if tag == "c":
            ax.legend(loc="upper right", fontsize=8, frameon=False)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGS, "fig5_xai_faithfulness.png"), bbox_inches="tight")
    plt.close()


def fig7_confusion():
    items = [
        ("(a) ResNet50 ensemble\nzero-shot, n = 691", os.path.join(LOGS, "lunghist700", "exp1_resnet50_main_metrics.json")),
        ("(b) ViT-S/16 ensemble\nzero-shot, n = 691", os.path.join(LOGS, "lunghist700", "exp1_vit_main_metrics.json")),
        (None, None),
        ("(c) ResNet50, ImageNet init.\nfine-tuned, OOF, n = 691", os.path.join(LOGS, "lunghist700", "e7_r50_metrics.json")),
        ("(d) ResNet50, LC25000 init.\nfine-tuned, OOF, n = 691", os.path.join(LOGS, "lunghist700", "e7_r50_lcinit_metrics.json")),
        ("(e) ViT-S/16, ImageNet init.\nfine-tuned, OOF, n = 691", os.path.join(LOGS, "lunghist700", "e7_vit_metrics.json")),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(11.5, 7.6))
    axes = axes.flatten()
    cls = ["ACA", "N", "SCC"]
    for ax, (ttl, path) in zip(axes, items):
        if path is None:
            ax.axis("off")
            continue
        cm = np.array(json.load(open(path, encoding="utf-8"))["confusion"])
        cm_pct = cm / cm.sum(axis=1, keepdims=True) * 100.0
        ax.imshow(cm_pct, cmap="Blues", vmin=0, vmax=100)
        for i in range(3):
            for j in range(3):
                col = "white" if cm_pct[i, j] > 55 else "#0F172A"
                ax.text(j, i, f"{cm[i, j]}\n{cm_pct[i, j]:.1f}%", ha="center", va="center", color=col, fontsize=9)
        ax.set_xticks(range(3))
        ax.set_yticks(range(3))
        ax.set_xticklabels(cls)
        ax.set_yticklabels(cls)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(ttl, fontsize=9.8)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGS, "fig7_lunghist700_confusion_matrices.png"), bbox_inches="tight")
    plt.close()


def fig8_roc():
    models = [
        ("ResNet50 (ImageNet)", os.path.join(LOGS, "lunghist700", "e7_r50_oof.csv"), BLUE, "-"),
        ("ResNet50 (LC25000)", os.path.join(LOGS, "lunghist700", "e7_r50_lcinit_oof.csv"), "#60A5FA", "--"),
        ("ViT-S/16 (ImageNet)", os.path.join(LOGS, "lunghist700", "e7_vit_oof.csv"), ORANGE, "-"),
    ]
    classes = [("(a) ACA vs rest", "ACA", "p_ACA"), ("(b) Normal vs rest", "N", "p_N"), ("(c) SCC vs rest", "SCC", "p_SCC")]
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.9), sharey=True)
    for ax, (ttl, cname, pcol) in zip(axes, classes):
        for lbl, fpath, col, ls in models:
            df = pd.read_csv(fpath)
            y_bin = (df["label"] == cname).astype(int)
            fpr, tpr, _ = roc_curve(y_bin, df[pcol])
            roc_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, color=col, ls=ls, lw=1.8, label=f"{lbl}, AUC = {roc_auc:.3f}")
        ax.plot([0, 1], [0, 1], ls=":", color=GREY, lw=1.2)
        ax.set_title(ttl, fontsize=10)
        ax.set_xlabel("False positive rate")
        ax.legend(loc="lower right", frameon=False, fontsize=8)
    axes[0].set_ylabel("True positive rate")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGS, "fig8_lunghist700_roc_curves.png"), bbox_inches="tight")
    plt.close()


def figS1_training_curves():
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 6.8))
    for row_idx, (sc, name, col) in enumerate([("resnet50_main", "ResNet50 (Adam 1e-4)", BLUE), ("vit_s_main", "ViT-S/16 (AdamW 1e-4)", ORANGE)]):
        for fold in range(5):
            h = pd.read_csv(os.path.join(LOGS, "runs", sc, f"fold_{fold}", "history.csv"))
            axes[row_idx, 0].plot(h["epoch"], h["val_loss"], alpha=0.75, lw=1.4, label=f"Fold {fold}")
            axes[row_idx, 1].plot(h["epoch"], h["val_acc"] * 100.0, alpha=0.75, lw=1.4, label=f"Fold {fold}")
        axes[row_idx, 0].set_title(f"{name} — Validation Loss", fontsize=10, color=col, fontweight="bold")
        axes[row_idx, 1].set_title(f"{name} — Validation Accuracy (%)", fontsize=10, color=col, fontweight="bold")
        axes[row_idx, 0].set_ylabel("Cross-Entropy Loss")
        axes[row_idx, 1].set_ylabel("Accuracy (%)")
        axes[row_idx, 0].legend(fontsize=7.5, frameon=False)
        axes[row_idx, 1].legend(fontsize=7.5, frameon=False)
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 1].set_xlabel("Epoch")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGS, "figS1_training_curves.png"), bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    fig2_scenarios()
    fig3_sensitivity()
    fig4_compute()
    fig5_faithfulness()
    fig7_confusion()
    fig8_roc()
    figS1_training_curves()
    print("[OK] Generated Figs 2, 3, 4, 5, 7, 8, and Fig S1 cleanly in figures/!")
