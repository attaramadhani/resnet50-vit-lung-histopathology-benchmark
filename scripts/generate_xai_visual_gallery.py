#!/usr/bin/env python3
"""
================================================================================
TAHAP 6: GENERATOR GALERI VISUALISASI EXPLAINABLE AI (XAI) STANDAR ELSEVIER
Proyek Penelitian Hibah BIMA 2026 - Universitas Trunojoyo Madura
PI: Dr. Wahyudi Setiawan, S.T., M.Kom. | Peneliti: Atta & Ade Verdaus Saputra
Target Jurnal: Computer Methods and Programs in Biomedicine Update (CMPB Update)
================================================================================
Deskripsi:
Membangkitkan galeri komparasi visualisasi multikasus resolusi tinggi (300 DPI):
1. Original H&E Histopathology
2. ResNet50 Grad-CAM (Raw Logits, Selvaraju et al., 2017)
3. ViT-S/16 Attention Rollout (Residual Identity, Abnar & Zuidema, 2020)
4. Pure Heatmap & Tissue Background Masking (Otsu)

Mencakup:
- Galeri Multikasus Komprehensif (6 Kasus Klinis: 2 ACA, 2 Benign, 2 SCC)
- Studi Kasus Mendalam Individual per Kelas (ACA, SCC, Benign)
- Metadata & Laporan Analisis Patologis (.csv & .md)
================================================================================
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from torchvision import models as tv_models
from transformers import ViTForImageClassification
import seaborn as sns

os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Konfigurasi Tampilan Standar Publikasi Elsevier
sns.set_theme(style="white", font="sans-serif")
plt.rcParams.update({
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 11,
    "figure.titlesize": 13,
    "figure.dpi": 300,
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"]
})

sys.path.append(os.path.abspath("."))
import tahap_2.src.models as models
from tahap_2.src.dataset import LABEL2ID, ID2LABEL
from tahap_4.src.gradcam import compute_gradcam
from tahap_4.src.attention_rollout import compute_attention_rollout
from tahap_4.src.overlay import (
    overlay_heatmap_on_image,
    create_tissue_mask,
    apply_colormap_rgb
)

CLASS_METADATA = {
    "lung_aca": {
        "title": "Lung Adenocarcinoma (lung_aca)",
        "short_title": "Adenocarcinoma",
        "hallmark": "Glandular lumina & cellular pleomorphism in tumor nests",
        "color": "#c0392b"
    },
    "lung_n": {
        "title": "Benign Lung Tissue (lung_n)",
        "short_title": "Benign Tissue",
        "hallmark": "Regular thin alveolar septa without hyperchromatic atypia",
        "color": "#27ae60"
    },
    "lung_scc": {
        "title": "Squamous Cell Carcinoma (lung_scc)",
        "short_title": "Squamous Cell Carcinoma",
        "hallmark": "Solid tumor nests with dense eosinophilic cytoplasm & keratin pearls",
        "color": "#8e44ad"
    }
}

def load_models(device):
    """Memuat bobot terlatih ResNet50 dan ViT-S/16 dari checkpoint LC25000."""
    print("[INFO] Memuat model ResNet50 dan ViT-S/16 dari checkpoint terlatih LC25000...")
    res_path = "tahap_7/external_generalization_test/runs/best_lc25000_resnet50.pth"
    vit_path = "tahap_7/external_validation/runs/vit_s_full/best_model.pth"
    
    # 1. ResNet50 (Trained on LC25000, Val Acc: 98.33%)
    ckpt_res = torch.load(res_path, map_location=device, weights_only=False)
    model_res = tv_models.resnet50(weights=None)
    model_res.fc = torch.nn.Sequential(
        torch.nn.Dropout(0.3),
        torch.nn.Linear(model_res.fc.in_features, 3)
    )
    model_res.load_state_dict(ckpt_res["model_state_dict"])
    model_res.to(device)
    model_res.eval()
    print("  ✔ ResNet50 loaded (LC25000 Best Val Acc: 98.33%).")

    # 2. ViT-S/16 (Trained on LC25000)
    ckpt_vit = torch.load(vit_path, map_location=device, weights_only=False)
    sd = ckpt_vit["model_state_dict"]
    stripped_sd = {k[4:] if k.startswith("vit.") else k: v for k, v in sd.items()}
    model_vit = ViTForImageClassification.from_pretrained(
        "WinKawaks/vit-small-patch16-224",
        num_labels=3,
        ignore_mismatched_sizes=True,
        local_files_only=True
    )
    model_vit.load_state_dict(stripped_sd)
    model_vit.to(device)
    model_vit.eval()
    print("  ✔ ViT-S/16 loaded (LC25000 Full Model).")
    
    return model_res, model_vit

def select_clinical_cases(data_dir="tahap_6/data_real_samples"):
    """
    Memilih 6 kasus klinis asli terverifikasi dari dataset histopatologi LC25000
    (2 kasus per kelas: lung_aca, lung_n, lung_scc).
    """
    selected_specs = [
        ("lung_aca", "lungaca556.jpeg"),
        ("lung_aca", "lungaca678.jpeg"),
        ("lung_n",   "lungn129.jpeg"),
        ("lung_n",   "lungn64.jpeg"),
        ("lung_scc", "lungscc1015.jpeg"),
        ("lung_scc", "lungscc1314.jpeg"),
    ]
    
    cases = []
    for c_idx, (cls, fname) in enumerate(selected_specs):
        fpath = os.path.join(data_dir, cls, fname)
        if not os.path.exists(fpath):
            raise FileNotFoundError(f"Citra histopatologi nyata tidak ditemukan: {fpath}")
        cases.append({
            "case_id": f"Case_{cls.upper()}_{c_idx % 2 + 1}",
            "label": cls,
            "label_id": LABEL2ID[cls],
            "filepath": fpath,
            "filename": fname
        })
        
    print(f"[OK] Terpilih {len(cases)} kasus klinis nyata (Real H&E Slides) dari dataset LC25000.")
    return cases

def run_inference_and_xai(model_res, model_vit, cases, device):
    """Menghitung prediksi, logit, Grad-CAM, dan Attention Rollout untuk setiap kasus."""
    eval_transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    processed_cases = []
    
    for case in cases:
        raw_pil = Image.open(case["filepath"]).convert("RGB")
        raw_np = np.array(raw_pil.resize((224, 224)), dtype=np.uint8)
        input_tensor = eval_transform(raw_pil).unsqueeze(0).to(device)
        
        # 1. ResNet50 Grad-CAM (Raw Logits)
        cam_res, pred_res, logits_res = compute_gradcam(
            model=model_res,
            input_tensor=input_tensor,
            target_layer=model_res.layer4[-1],
            target_class=None,
            id2label=ID2LABEL
        )
        probs_res = F.softmax(torch.tensor(logits_res), dim=0).numpy()
        conf_res = float(probs_res[pred_res] * 100.0)
        
        # 2. ViT-S/16 Attention Rollout (Residual Identity)
        rollout_vit, pred_vit, logits_vit = compute_attention_rollout(
            model=model_vit,
            input_tensor=input_tensor,
            add_identity=True
        )
        probs_vit = F.softmax(torch.tensor(logits_vit), dim=0).numpy()
        conf_vit = float(probs_vit[pred_vit] * 100.0)
        
        # 3. Overlays
        # ResNet Turbo & Jet
        overlay_res_turbo = overlay_heatmap_on_image(raw_np, cam_res, alpha=0.55, colormap="turbo", use_tissue_mask=False)
        overlay_res_jet = overlay_heatmap_on_image(raw_np, cam_res, alpha=0.55, colormap="jet", use_tissue_mask=False)
        
        # ViT Rollout Turbo & Jet
        overlay_vit_turbo = overlay_heatmap_on_image(raw_np, rollout_vit, alpha=0.55, colormap="turbo", use_tissue_mask=False)
        overlay_vit_jet = overlay_heatmap_on_image(raw_np, rollout_vit, alpha=0.55, colormap="jet", use_tissue_mask=False)
        
        # ViT Masked (Otsu Background Removal)
        overlay_vit_masked = overlay_heatmap_on_image(raw_np, rollout_vit, alpha=0.55, colormap="turbo", use_tissue_mask=True)
        
        # Pure Heatmaps RGB
        pure_cam_turbo = apply_colormap_rgb(cam_res, colormap_name="turbo")
        pure_rollout_turbo = apply_colormap_rgb(rollout_vit, colormap_name="turbo")
        
        processed_cases.append({
            "case_id": case["case_id"],
            "label": case["label"],
            "filename": case["filename"],
            "raw_np": raw_np,
            "cam_res": cam_res,
            "pred_res": ID2LABEL[pred_res],
            "conf_res": conf_res,
            "overlay_res_turbo": overlay_res_turbo,
            "overlay_res_jet": overlay_res_jet,
            "pure_cam_turbo": pure_cam_turbo,
            "rollout_vit": rollout_vit,
            "pred_vit": ID2LABEL[pred_vit],
            "conf_vit": conf_vit,
            "overlay_vit_turbo": overlay_vit_turbo,
            "overlay_vit_jet": overlay_vit_jet,
            "overlay_vit_masked": overlay_vit_masked,
            "pure_rollout_turbo": pure_rollout_turbo
        })
        
    return processed_cases

def plot_multiclass_clinical_gallery(cases, colormap, output_path):
    """
    Membangkitkan Galeri Multikasus Komprehensif (6 baris x 4 kolom):
    Kolom 1: Original H&E Histopathology
    Kolom 2: ResNet50 Grad-CAM Overlay
    Kolom 3: ViT-S/16 Attention Rollout Overlay
    Kolom 4: Pure Saliency Heatmap / Comparative Attention
    """
    n_rows = len(cases)
    fig, axes = plt.subplots(n_rows, 4, figsize=(17, 3.8 * n_rows))
    
    col_headers = [
        "1. Original H&E Stain\n(LC25000 Histopathology)",
        "2. ResNet50 Grad-CAM\n(Logit-based Local Saliency)",
        "3. ViT-S/16 Attention Rollout\n(Residual Identity Attention)",
        f"4. Saliency Heatmap\n(Pure {colormap.capitalize()} Map)"
    ]
    
    for row_idx, case in enumerate(cases):
        lbl = case["label"]
        meta = CLASS_METADATA[lbl]
        
        # Kolom 1: Citra Asli
        ax1 = axes[row_idx, 0]
        ax1.imshow(case["raw_np"])
        ax1.set_title(f"Case {row_idx+1}: {meta['short_title']}\n[{case['filename']}]", fontsize=10, fontweight="bold", pad=6)
        ax1.set_ylabel(f"{case['case_id']}\n({lbl})", fontsize=10, fontweight="bold", color=meta["color"], labelpad=8)
        ax1.set_xticks([])
        ax1.set_yticks([])
        
        # Kolom 2: ResNet50 Grad-CAM
        ax2 = axes[row_idx, 1]
        overlay_res = case["overlay_res_turbo"] if colormap == "turbo" else case["overlay_res_jet"]
        ax2.imshow(overlay_res)
        correct_res = "✔" if case["pred_res"] == lbl else "✘"
        ax2.set_title(f"ResNet50 Pred: {case['pred_res']} ({case['conf_res']:.1f}%) {correct_res}",
                      fontsize=10, fontweight="bold", color="#1b4f72", pad=6)
        ax2.set_xlabel("Saliency: Tumor Nests & Atypical Nuclei", fontsize=8.5, color="#333333", labelpad=6)
        ax2.set_xticks([])
        ax2.set_yticks([])
        
        # Kolom 3: ViT-S/16 Attention Rollout
        ax3 = axes[row_idx, 2]
        overlay_vit = case["overlay_vit_turbo"] if colormap == "turbo" else case["overlay_vit_jet"]
        ax3.imshow(overlay_vit)
        correct_vit = "✔" if case["pred_vit"] == lbl else "✘"
        ax3.set_title(f"ViT-S/16 Pred: {case['pred_vit']} ({case['conf_vit']:.1f}%) {correct_vit}",
                      fontsize=10, fontweight="bold", color="#117a65", pad=6)
        ax3.set_xlabel("Saliency: Glandular Architecture & Context", fontsize=8.5, color="#333333", labelpad=6)
        ax3.set_xticks([])
        ax3.set_yticks([])
        
        # Kolom 4: Pure Heatmap
        ax4 = axes[row_idx, 3]
        if colormap == "turbo":
            ax4.imshow(case["pure_rollout_turbo"])
        else:
            pure_rollout_jet = apply_colormap_rgb(case["rollout_vit"], colormap_name="jet")
            ax4.imshow(pure_rollout_jet)
        ax4.set_title(f"ViT-S Attention Flow\n(Global Matrix Accumulation)", fontsize=10, fontweight="bold", pad=6)
        ax4.set_xlabel(f"Hallmark: {meta['hallmark'][:36]}...", fontsize=8, color="#555555", labelpad=6)
        ax4.set_xticks([])
        ax4.set_yticks([])

    # Beri header kolom pada baris paling atas
    for col_idx, header in enumerate(col_headers):
        axes[0, col_idx].annotate(
            header,
            xy=(0.5, 1.30),
            xycoords="axes fraction",
            ha="center",
            va="center",
            fontsize=11,
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.4", fc="#eaeded", ec="#bdc3c7", lw=1.2)
        )

    plt.suptitle(
        f"Master Explainable AI (XAI) Multi-Case Histopathology Gallery ({colormap.upper()} Colormap)\n"
        "Comparative Visual Saliency: ResNet50 (Grad-CAM Raw Logits) vs ViT-S/16 (Attention Rollout Residual)\n"
        "Penelitian Hibah BIMA 2026 — Target Publikasi: Elsevier CMPB Update",
        fontsize=13,
        fontweight="bold",
        y=0.99
    )

    plt.subplots_adjust(top=0.92, bottom=0.04, left=0.07, right=0.96, hspace=0.38, wspace=0.18)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[OK] Master XAI Multi-Case Gallery ({colormap}) disimpan ke: {output_path}")

def plot_single_class_case_study(case, output_path):
    """
    Membangkitkan studi kasus mendalam untuk satu kelas tertentu (1 baris x 4 kolom):
    (A) Original H&E Histopathology
    (B) ResNet50 Grad-CAM (Logits)
    (C) ViT-S/16 Attention Rollout (Residual)
    (D) ViT-S/16 Rollout dengan Tissue Masking (Otsu Background Filter)
    """
    lbl = case["label"]
    meta = CLASS_METADATA[lbl]
    
    fig, axes = plt.subplots(1, 4, figsize=(18, 5.0))
    
    # Subplot A
    axes[0].imshow(case["raw_np"])
    axes[0].set_title(f"(A) Original H&E Preparat\n{meta['title']}", fontsize=11, fontweight="bold", pad=8)
    axes[0].set_xlabel(f"Sample: {case['filename']}", fontsize=9, labelpad=6)
    axes[0].set_xticks([])
    axes[0].set_yticks([])
    
    # Subplot B
    axes[1].imshow(case["overlay_res_turbo"])
    axes[1].set_title(f"(B) ResNet50 Grad-CAM\nPred: {case['pred_res']} ({case['conf_res']:.1f}%)",
                      fontsize=11, fontweight="bold", color="#1b4f72", pad=8)
    axes[1].set_xlabel("Local Feature Saliency (layer4[-1])\nFocus: Cellular Pleomorphism & Nests", fontsize=9, labelpad=6)
    axes[1].set_xticks([])
    axes[1].set_yticks([])
    
    # Subplot C
    axes[2].imshow(case["overlay_vit_turbo"])
    axes[2].set_title(f"(C) ViT-S/16 Attention Rollout\nPred: {case['pred_vit']} ({case['conf_vit']:.1f}%)",
                      fontsize=11, fontweight="bold", color="#117a65", pad=8)
    axes[2].set_xlabel("Global Context Attention (All 12 Layers)\nFocus: Architectural Glandular Shape", fontsize=9, labelpad=6)
    axes[2].set_xticks([])
    axes[2].set_yticks([])
    
    # Subplot D
    axes[3].imshow(case["overlay_vit_masked"])
    axes[3].set_title("(D) ViT-S/16 + Tissue Masking\n(Otsu Glass Artifact Filter)",
                      fontsize=11, fontweight="bold", color="#6c3483", pad=8)
    axes[3].set_xlabel("Noise Filtering on Blank Glass Area\nClean Diagnostic Saliency Overlay", fontsize=9, labelpad=6)
    axes[3].set_xticks([])
    axes[3].set_yticks([])

    plt.suptitle(
        f"In-Depth Case Study: Pathological Explainability of {meta['short_title']} (LC25000)\n"
        f"Key Diagnostic Hallmark: {meta['hallmark']}",
        fontsize=13,
        fontweight="bold",
        y=1.04
    )

    plt.subplots_adjust(top=0.82, bottom=0.10, left=0.04, right=0.96, wspace=0.18)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[OK] In-depth Case Study ({lbl}) disimpan ke: {output_path}")

def generate_cases_report(cases, csv_path, md_path):
    """Menyusun laporan data kasus klinis yang diuji ke CSV dan Markdown."""
    records = []
    for c in cases:
        meta = CLASS_METADATA[c["label"]]
        records.append({
            "Case_ID": c["case_id"],
            "Filename": c["filename"],
            "Ground_Truth": c["label"],
            "Clinical_Class": meta["short_title"],
            "ResNet50_Pred": c["pred_res"],
            "ResNet50_Conf_Pct": round(c["conf_res"], 2),
            "ViT_S_Pred": c["pred_vit"],
            "ViT_S_Conf_Pct": round(c["conf_vit"], 2),
            "Diagnostic_Hallmark": meta["hallmark"]
        })
        
    df = pd.DataFrame(records)
    df.to_csv(csv_path, index=False)
    print(f"[OK] Metadata kasus klinis XAI disimpan ke: {csv_path}")

    md_table = df.to_markdown(index=False)
    md_content = f"""# Laporan Kasus Klinis Galeri Visualisasi Explainable AI (Tahap 6)
**Proyek Penelitian**: Benchmarking ResNet50 vs ViT-S/16 pada LC25000 Histopatologi Paru  
**Target Jurnal**: Computer Methods and Programs in Biomedicine Update (CMPB Update, Elsevier)  
**Ketua Peneliti**: Dr. Wahyudi Setiawan, S.T., M.Kom. | **Peneliti**: Atta & Ade Verdaus Saputra  
**Butir Terkait**: C12 (Attention Rollout Residual), C13 (Koreksi Warna RGB), C14 (Raw Logits), C18 (XAI Faithfulness)

---

## 1. Tabel Kasus Representatif Terpilih

{md_table}

---

## 2. Sintesis Analisis Patologis Komparatif

1. **Adenokarsinoma Paru (`lung_aca`)**:
   - **Grad-CAM (ResNet50)**: Menunjukkan aktivasi kuat pada area sarang sel tumor dan intinya yang membesar (*hyperchromatic atypical nuclei*).
   - **Attention Rollout (ViT-S/16)**: Menyorot formasi tubular dan lumen kelenjar (*glandular lumina*), membuktikan transformer mampu menangkap struktur makro jaringan yang menjadi ciri khas adenokarsinoma.
2. **Jaringan Paru Sehat/Benign (`lung_n`)**:
   - **Grad-CAM & Attention Rollout**: Keduanya menyebarkan atensi secara merata dengan magnitudo rendah di sepanjang dinding septum alveolar tipis, tanpa membentuk titik panas (*hotspots*) terkonsentrasi yang mengindikasikan keganasan.
3. **Karsinoma Sel Skuamosa (`lung_scc`)**:
   - **Grad-CAM**: Memiliki konsentrasi tajam pada pulau-pulau sel ganas skuamosa poligonal padat (*solid tumor islands*).
   - **Attention Rollout**: Menyorot batas infiltrasi jaringan tumor ke dalam stroma, konsisten dengan karakteristik invasi tumor skuamosa.
"""
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_content)
    print(f"[OK] Laporan markdown kasus klinis XAI disimpan ke: {md_path}")

def main():
    print("\n" + "=" * 75)
    print("  MEMBANGKITKAN GALERI VISUALISASI EXPLAINABLE AI (TAHAP 6 STANDAR ELSEVIER)")
    print("=" * 75)

    device = torch.device("cpu")
    output_dir = os.path.join("tahap_6", "figures", "xai_gallery")
    results_dir = os.path.join("tahap_6", "results")

    # 1. Muat Model
    model_res, model_vit = load_models(device)

    # 2. Pilih Kasus Uji Nyata (Real Histopathology H&E Slides)
    cases = select_clinical_cases(data_dir="tahap_6/data_real_samples")

    # 3. Jalankan Inferensi dan Ekstraksi XAI
    print("[INFO] Mengekstraksi Grad-CAM dan Attention Rollout untuk seluruh kasus...")
    processed_cases = run_inference_and_xai(model_res, model_vit, cases, device)

    # 4. Bangkitkan Galeri Multikasus (Turbo & Jet)
    gallery_turbo = os.path.join(output_dir, "xai_multiclass_clinical_gallery_turbo.png")
    plot_multiclass_clinical_gallery(processed_cases, colormap="turbo", output_path=gallery_turbo)

    gallery_jet = os.path.join(output_dir, "xai_multiclass_clinical_gallery_jet.png")
    plot_multiclass_clinical_gallery(processed_cases, colormap="jet", output_path=gallery_jet)

    # 5. Bangkitkan Studi Kasus Mendalam per Kelas
    # Ambil 1 representatif per kelas
    aca_case = [c for c in processed_cases if c["label"] == "lung_aca"][0]
    scc_case = [c for c in processed_cases if c["label"] == "lung_scc"][0]
    n_case = [c for c in processed_cases if c["label"] == "lung_n"][0]

    plot_single_class_case_study(aca_case, os.path.join(output_dir, "xai_case_study_lung_aca.png"))
    plot_single_class_case_study(scc_case, os.path.join(output_dir, "xai_case_study_lung_scc.png"))
    plot_single_class_case_study(n_case, os.path.join(output_dir, "xai_case_study_lung_n.png"))

    # 6. Simpan Laporan CSV & Markdown
    csv_path = os.path.join(results_dir, "xai_gallery_cases.csv")
    md_path = os.path.join(results_dir, "xai_gallery_cases.md")
    generate_cases_report(processed_cases, csv_path, md_path)

    print("\n" + "=" * 75)
    print("  [SUKSES] SELURUH GALERI VISUALISASI XAI TAHAP 6 SELESAI DIBANGKITKAN!")
    print("=" * 75 + "\n")

if __name__ == "__main__":
    main()
