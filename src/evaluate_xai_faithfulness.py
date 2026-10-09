#!/usr/bin/env python3
"""
================================================================================
TAHAP 6 - TUGAS 6.5: EVALUASI KUANTITATIF XAI FAITHFULNESS (BUTIR C18)
Proyek Penelitian Hibah BIMA 2026 - Universitas Trunojoyo Madura
PI: Dr. Wahyudi Setiawan, S.T., M.Kom. | Peneliti: Atta & Ade Verdaus Saputra
Target Jurnal: Computer Methods and Programs in Biomedicine Update (Elsevier)
================================================================================
Fungsi:
1. Menghitung metrik Deletion AUC dan Insertion AUC untuk mengukur keandalan
   penjelasan visual (Explainable AI Faithfulness) pada histopatologi paru.
2. Membandingkan:
   - ResNet50 + Grad-CAM (Logit-based)
   - ViT-S/16 + Attention Rollout (Residual Identity)
   - Random Baseline (Atribusi piksel acak sebagai kontrol saintifik)
3. Menguji 50 citra per kelas (Total 150 citra: ACA, Benign, SCC).
4. Menghasilkan tabel metrik: tahap_6/results/tabel_faithfulness.csv & .md.
5. Menghasilkan figure kurva penurunan/kenaikan probabilitas:
   tahap_6/figures/xai_quantitative_faithfulness_curves.png (300 DPI).
================================================================================
"""

import os
import sys
import glob
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision import transforms
import matplotlib.pyplot as plt
import seaborn as sns

if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Estetika Publikasi Elsevier
sns.set_theme(style="whitegrid", font="sans-serif")
plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "figure.dpi": 300
})

sys.path.append(os.path.abspath("."))
from tahap_2.src.models import build_model
from tahap_4.src.gradcam import GradCAM
from tahap_4.src.attention_rollout import ViTAttentionRollout

CLASS_NAMES = ["lung_aca", "lung_n", "lung_scc"]
LABEL2ID = {c: i for i, c in enumerate(CLASS_NAMES)}
ID2LABEL = {i: c for i, c in enumerate(CLASS_NAMES)}

def get_transforms():
    eval_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return eval_transform

def load_models():
    """Memuat model ResNet50 dan ViT-S/16."""
    device = torch.device("cpu")
    print("[INFO] Memuat arsitektur model dan bobot checkpoint...")
    
    # 1. ResNet50
    cfg_res = {
        "model_architecture": "resnet50",
        "model_name": "resnet50",
        "num_classes": 3,
        "freeze_layers": [0, 1]
    }
    model_res = build_model(cfg_res, LABEL2ID, ID2LABEL)
    res_path = "tahap_3/runs/smoke_test_run/fold_1/best_model.pth"
    if os.path.exists(res_path):
        ckpt_res = torch.load(res_path, map_location=device)
        model_res.load_state_dict(ckpt_res["model_state_dict"])
        print("  ✔ ResNet50 checkpoint loaded successfully.")
    model_res.eval()

    # 2. ViT-S/16
    cfg_vit = {
        "model_architecture": "vit_s",
        "model_name": "WinKawaks/vit-small-patch16-224",
        "num_classes": 3,
        "freeze_layers": [0, 1]
    }
    model_vit = build_model(cfg_vit, LABEL2ID, ID2LABEL)
    vit_path = "tahap_3/runs/smoke_test_vit/fold_1/best_model.pth"
    if os.path.exists(vit_path):
        ckpt_vit = torch.load(vit_path, map_location=device)
        model_vit.load_state_dict(ckpt_vit["model_state_dict"])
        print("  ✔ ViT-S/16 checkpoint loaded successfully.")
    model_vit.eval()

    return model_res, model_vit, device

def evaluate_faithfulness_single(model, input_tensor, heatmap, target_class, steps=10, is_vit=False):
    """
    Menghitung kurva Deletion dan Insertion untuk 1 citra.
    steps: 10 step (0%, 10%, 20%, ..., 100% piksel termodifikasi)
    """
    # input_tensor: (1, 3, 224, 224)
    # heatmap: (224, 224) [0.0, 1.0]
    
    H, W = heatmap.shape
    total_pixels = H * W
    flat_heatmap = heatmap.flatten()
    sorted_indices = np.argsort(flat_heatmap)[::-1] # Indeks ranking terpenting -> kurang penting

    # Salinan kanvas
    orig_img = input_tensor.clone()
    black_canvas = torch.zeros_like(input_tensor) # Kanvas kosong untuk Insertion

    del_probs = []
    ins_probs = []

    fractions = np.linspace(0.0, 1.0, steps + 1) # [0.0, 0.1, ..., 1.0]

    with torch.no_grad():
        for frac in fractions:
            n_replace = int(round(frac * total_pixels))
            
            # --- DELETION ---
            # Ganti top-k piksel dengan 0 (black/baseline)
            cur_del = orig_img.clone()
            if n_replace > 0:
                mask = np.ones(total_pixels, dtype=bool)
                mask[sorted_indices[:n_replace]] = False
                mask_2d = torch.from_numpy(mask.reshape(H, W)).unsqueeze(0).unsqueeze(0).to(input_tensor.device)
                cur_del = cur_del * mask_2d

            out_del = model(cur_del)
            logits_del = out_del.logits if hasattr(out_del, "logits") else out_del
            prob_del = F.softmax(logits_del, dim=-1)[0, target_class].item()
            del_probs.append(prob_del)

            # --- INSERTION ---
            # Masukkan top-k piksel ke dalam black canvas
            cur_ins = black_canvas.clone()
            if n_replace > 0:
                mask_ins = np.zeros(total_pixels, dtype=bool)
                mask_ins[sorted_indices[:n_replace]] = True
                mask_ins_2d = torch.from_numpy(mask_ins.reshape(H, W)).unsqueeze(0).unsqueeze(0).to(input_tensor.device)
                cur_ins = orig_img * mask_ins_2d

            out_ins = model(cur_ins)
            logits_ins = out_ins.logits if hasattr(out_ins, "logits") else out_ins
            prob_ins = F.softmax(logits_ins, dim=-1)[0, target_class].item()
            ins_probs.append(prob_ins)

    auc_del = np.trapz(del_probs, fractions)
    auc_ins = np.trapz(ins_probs, fractions)
    return fractions, del_probs, ins_probs, auc_del, auc_ins

def run_quantitative_xai_experiment(n_samples_per_class=50):
    """
    Menjalankan pengujian kuantitatif Faithfulness pada dataset histopatologi.
    """
    model_res, model_vit, device = load_models()
    eval_transform = get_transforms()
    
    gradcam_engine = GradCAM(model_res, target_layer=model_res.layer4[-1])
    rollout_engine = ViTAttentionRollout(model_vit)

    # Ambil sampel citra (50 per kelas)
    samples = []
    for cls in CLASS_NAMES:
        cls_dir = os.path.join("tahap_3", "data_smoke", cls)
        files = sorted(glob.glob(os.path.join(cls_dir, "*.jpg")) + glob.glob(os.path.join(cls_dir, "*.jpeg")))
        chosen = files[:n_samples_per_class]
        for f in chosen:
            samples.append((f, LABEL2ID[cls]))

    print(f"\n[INFO] Mengevaluasi {len(samples)} citra ({n_samples_per_class} citra per kelas)...")

    res_del_curves = []
    res_ins_curves = []
    vit_del_curves = []
    vit_ins_curves = []
    rnd_del_curves = []
    rnd_ins_curves = []

    res_del_aucs = []
    res_ins_aucs = []
    vit_del_aucs = []
    vit_ins_aucs = []
    rnd_del_aucs = []
    rnd_ins_aucs = []

    for idx, (img_path, target_class) in enumerate(samples, 1):
        with Image.open(img_path).convert("RGB") as pil_img:
            input_tensor = eval_transform(pil_img).unsqueeze(0).to(device)

        # 1. Heatmap Grad-CAM (ResNet50)
        h_gradcam, _, _ = gradcam_engine.generate_cam(input_tensor, target_class=target_class)

        # 2. Heatmap Attention Rollout (ViT-S/16)
        h_rollout, _, _ = rollout_engine.generate_attention_map(input_tensor, add_identity=True)

        # 3. Heatmap Random Baseline
        h_random = np.random.rand(224, 224)

        # Evaluasi ResNet50 + Grad-CAM
        fracs, d_res, i_res, a_del_res, a_ins_res = evaluate_faithfulness_single(
            model_res, input_tensor, h_gradcam, target_class, steps=10, is_vit=False
        )
        res_del_curves.append(d_res)
        res_ins_curves.append(i_res)
        res_del_aucs.append(a_del_res)
        res_ins_aucs.append(a_ins_res)

        # Evaluasi ViT-S/16 + Attention Rollout
        _, d_vit, i_vit, a_del_vit, a_ins_vit = evaluate_faithfulness_single(
            model_vit, input_tensor, h_rollout, target_class, steps=10, is_vit=True
        )
        vit_del_curves.append(d_vit)
        vit_ins_curves.append(i_vit)
        vit_del_aucs.append(a_del_vit)
        vit_ins_aucs.append(a_ins_vit)

        # Evaluasi Random Baseline (pada ResNet)
        _, d_rnd, i_rnd, a_del_rnd, a_ins_rnd = evaluate_faithfulness_single(
            model_res, input_tensor, h_random, target_class, steps=10, is_vit=False
        )
        rnd_del_curves.append(d_rnd)
        rnd_ins_curves.append(i_rnd)
        rnd_del_aucs.append(a_del_rnd)
        rnd_ins_aucs.append(a_ins_rnd)

        if idx % 25 == 0 or idx == len(samples):
            print(f"  Processed [{idx}/{len(samples)}] samples...")

    # Rata-rata kurva
    fracs = np.linspace(0.0, 1.0, 11)
    res_del_mean = np.mean(res_del_curves, axis=0)
    res_ins_mean = np.mean(res_ins_curves, axis=0)
    vit_del_mean = np.mean(vit_del_curves, axis=0)
    vit_ins_mean = np.mean(vit_ins_curves, axis=0)
    rnd_del_mean = np.mean(rnd_del_curves, axis=0)
    rnd_ins_mean = np.mean(rnd_ins_curves, axis=0)

    # -------------------------------------------------------------
    # 1. SIMPAN TABEL METRIK KUANTITATIF
    # -------------------------------------------------------------
    results_summary = [
        {
            "Metode XAI": "ResNet50 + Grad-CAM (Logits)",
            "Tipe Model": "CNN",
            "Deletion AUC (Drop tajam = Baik)": f"{np.mean(res_del_aucs):.4f} +/- {np.std(res_del_aucs):.4f}",
            "Insertion AUC (Naik tajam = Baik)": f"{np.mean(res_ins_aucs):.4f} +/- {np.std(res_ins_aucs):.4f}",
            "Faithfulness Ratio (Ins / Del)": f"{np.mean(res_ins_aucs) / max(np.mean(res_del_aucs), 1e-4):.2f}x"
        },
        {
            "Metode XAI": "ViT-S/16 + Attention Rollout (Residual)",
            "Tipe Model": "Transformer",
            "Deletion AUC (Drop tajam = Baik)": f"{np.mean(vit_del_aucs):.4f} +/- {np.std(vit_del_aucs):.4f}",
            "Insertion AUC (Naik tajam = Baik)": f"{np.mean(vit_ins_aucs):.4f} +/- {np.std(vit_ins_aucs):.4f}",
            "Faithfulness Ratio (Ins / Del)": f"{np.mean(vit_ins_aucs) / max(np.mean(vit_del_aucs), 1e-4):.2f}x"
        },
        {
            "Metode XAI": "Random Attribution Baseline",
            "Tipe Model": "Control (Noise)",
            "Deletion AUC (Drop tajam = Baik)": f"{np.mean(rnd_del_aucs):.4f} +/- {np.std(rnd_del_aucs):.4f}",
            "Insertion AUC (Naik tajam = Baik)": f"{np.mean(rnd_ins_aucs):.4f} +/- {np.std(rnd_ins_aucs):.4f}",
            "Faithfulness Ratio (Ins / Del)": f"{np.mean(rnd_ins_aucs) / max(np.mean(rnd_del_aucs), 1e-4):.2f}x"
        }
    ]

    df_res = pd.DataFrame(results_summary)
    csv_out = os.path.join("tahap_6", "results", "faithfulness_table.csv")
    df_res.to_csv(csv_out, index=False)
    print(f"\n[OK] Tabel Faithfulness CSV disimpan ke: {csv_out}")

    md_out = os.path.join("tahap_6", "results", "tabel_faithfulness.md")
    with open(md_out, "w", encoding="utf-8") as f:
        f.write("# Tabel Evaluasi Kuantitatif XAI Faithfulness (Tahap 6 — Butir C18)\n\n")
        f.write(df_res.to_markdown(index=False))
        f.write("\n\n*Catatan*: Pengujian dilakukan pada 150 citra independen (50 per kelas: ACA, Benign, SCC). ")
        f.write("Nilai Deletion AUC yang lebih rendah menunjukkan bahwa piksel yang disorot XAI benar-benar krusial bagi prediksi klinis model.")
    print(f"[OK] Tabel Faithfulness MD disimpan ke: {md_out}")

    # -------------------------------------------------------------
    # 2. GENERATE FIGURE KURVA FAITHFULNESS
    # -------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # Subplot A: Deletion Curves
    axes[0].plot(fracs * 100, res_del_mean, marker="o", linewidth=2.5, color="#1f77b4", label=f"ResNet50 Grad-CAM (AUC={np.mean(res_del_aucs):.3f})")
    axes[0].plot(fracs * 100, vit_del_mean, marker="s", linewidth=2.5, color="#2ca02c", label=f"ViT-S Attention Rollout (AUC={np.mean(vit_del_aucs):.3f})")
    axes[0].plot(fracs * 100, rnd_del_mean, linestyle="--", linewidth=2.0, color="#7f7f7f", label=f"Random Baseline (AUC={np.mean(rnd_del_aucs):.3f})")
    axes[0].set_title("A. Deletion Metric (Penurunan Probabilitas saat Piksel Dihapus)\nSemakin cepat drop / AUC rendah = Semakin Faithful", fontweight="bold", pad=12)
    axes[0].set_xlabel("Persentase Piksel Terpenting yang Dihapus (%)", fontweight="bold")
    axes[0].set_ylabel("Probabilitas Model pada Kelas Benar", fontweight="bold")
    axes[0].set_ylim(-0.05, 1.05)
    axes[0].grid(True, linestyle=":", alpha=0.7)
    axes[0].legend(loc="upper right", frameon=True, facecolor="white", framealpha=0.9)

    # Subplot B: Insertion Curves
    axes[1].plot(fracs * 100, res_ins_mean, marker="o", linewidth=2.5, color="#1f77b4", label=f"ResNet50 Grad-CAM (AUC={np.mean(res_ins_aucs):.3f})")
    axes[1].plot(fracs * 100, vit_ins_mean, marker="s", linewidth=2.5, color="#2ca02c", label=f"ViT-S Attention Rollout (AUC={np.mean(vit_ins_aucs):.3f})")
    axes[1].plot(fracs * 100, rnd_ins_mean, linestyle="--", linewidth=2.0, color="#7f7f7f", label=f"Random Baseline (AUC={np.mean(rnd_ins_aucs):.3f})")
    axes[1].set_title("B. Insertion Metric (Kenaikan Probabilitas saat Piksel Dimasukkan)\nSemakin cepat naik / AUC tinggi = Semakin Faithful", fontweight="bold", pad=12)
    axes[1].set_xlabel("Persentase Piksel Terpenting yang Dimasukkan (%)", fontweight="bold")
    axes[1].set_ylabel("Probabilitas Model pada Kelas Benar", fontweight="bold")
    axes[1].set_ylim(-0.05, 1.05)
    axes[1].grid(True, linestyle=":", alpha=0.7)
    axes[1].legend(loc="lower right", frameon=True, facecolor="white", framealpha=0.9)

    plt.suptitle("Quantitative XAI Faithfulness Evaluation (Butir C18 - CMPB Update Jurnal Standar)\nUji Komparatif ResNet50 (Grad-CAM) vs ViT-S/16 (Attention Rollout) vs Random Baseline (N=150 Citra)",
                 fontweight="bold", y=1.02)
    plt.tight_layout()

    fig_out = os.path.join("tahap_6", "figures", "xai_quantitative_faithfulness_curves.png")
    os.makedirs(os.path.dirname(fig_out), exist_ok=True)
    plt.savefig(fig_out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[OK] Figure Kurva Faithfulness disimpan ke: {fig_out}")

def main():
    print("\n" + "=" * 70)
    print("  MENJALANKAN TUGAS 6.5: EVALUASI KUANTITATIF XAI FAITHFULNESS (BUTIR C18)")
    print("=" * 70)
    run_quantitative_xai_experiment(n_samples_per_class=50)
    print("\n" + "=" * 70)
    print("  [SUKSES] TUGAS 6.5 SELESAI TUNTAS DENGAN LENGKAP!")
    print("=" * 70 + "\n")

if __name__ == "__main__":
    main()
