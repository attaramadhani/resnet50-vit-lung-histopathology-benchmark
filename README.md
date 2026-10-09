# Beyond the LC25000 Ceiling: Optimizer Robustness and External Validity of a Convolutional Network and a Vision Transformer for Lung Cancer Histopathology Classification

[![Python](https://img.shields.io/badge/python-3.11%20%7C%203.12-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2%2B-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23271475.svg)](https://doi.org/10.5281/zenodo.23271475)
[![Target Journal](https://img.shields.io/badge/Elsevier-CMPB_Update-orange.svg)](https://www.sciencedirect.com/journal/computer-methods-and-programs-in-biomedicine-update)

Official open-science replication package for the research article submitted to Elsevier's *Computer Methods and Programs in Biomedicine Update* (CMPB Update).

---

## 👥 Authors & Affiliations

- **Wahyudi Setiawan** (Corresponding author: `wsetiawan@trunojoyo.ac.id`, ORCID: [0000-0003-0068-1443](https://orcid.org/0000-0003-0068-1443))
- **Attala Alif Ramadhani Tri Hida**
- **Ade Verdaus Saputra**

*Department of Information Systems, Faculty of Engineering, Universitas Trunojoyo Madura, Bangkalan 69162, East Java, Indonesia*

---

## 📌 Replication Package Overview

In accordance with the *Checklist for Artificial Intelligence in Medical Imaging* (CLAIM 2024 Update) and the Data Availability statement of the manuscript, this repository provides:
1. **Source Code (`src/`)**: Clean PyTorch implementations for model architectures (ResNet50 & ViT-S/16), stratified 5-fold cross-validation training, metric evaluation, patient-grouped cross-validation, and quantitative XAI perturbation testing.
2. **Split Manifests (`data_splits/`)**: Deterministic split files for the LC25000 benchmark, audited perceptual hashing groups, and patient-grouped 5-fold cross-validation on the clinical LungHist700 cohort.
3. **Configuration Files (`configs/`)**: Exact YAML configuration files for all 10 experimental scenarios.
4. **Per-Epoch Telemetry (`logs_and_metrics/`)**: Complete fold-by-fold epoch progression histories (`history.csv`) capturing training/validation loss, accuracy, and learning rate trajectories across all runs.
5. **Per-Fold & Benchmark Metrics (`logs_and_metrics/`)**: Detailed evaluation JSON metrics and cross-validation summaries (Accuracy, Macro F1, Precision, Recall, AUC-ROC, Out-of-Fold predictions).
6. **Publication Figure Scripts (`scripts/`) & 300 DPI Outputs (`figures/`)**: Automated reproduction scripts to generate high-resolution ROC curves, confusion matrices, XAI faithfulness perturbation curves, and visual heatmaps.

---

## 🧪 Experimental Design & Scenarios

### 1. Controlled 10-Scenario Grid (LC25000 Benchmark)
Comparative benchmarking of **ResNet50** (25.6M parameters) and **ViT-S/16** (22.1M parameters) under identical 5-fold cross-validation:

| Scenario | Model | Optimizer | Base Learning Rate | Weight Decay | Scheduler |
|:---:|:---:|:---:|:---:|:---:|:---:|
| **S1 (Main)** | ResNet50 | AdamW | $1 \times 10^{-4}$ | $1 \times 10^{-2}$ | Cosine Annealing |
| **S2** | ResNet50 | SGD + Momentum (0.9) | $1 \times 10^{-2}$ | $1 \times 10^{-4}$ | StepLR |
| **S3** | ResNet50 | RMSprop (alpha=0.99) | $1 \times 10^{-4}$ | 0.0 | Cosine Annealing |
| **S4** | ResNet50 | AdamW (High LR) | $1 \times 10^{-3}$ | $1 \times 10^{-2}$ | Cosine Annealing |
| **S5** | ResNet50 | AdamW (Low LR) | $1 \times 10^{-5}$ | $1 \times 10^{-2}$ | Cosine Annealing |
| **S6 (Main)** | ViT-S/16 | AdamW | $1 \times 10^{-4}$ | $5 \times 10^{-2}$ | Linear Warmup + Cosine |
| **S7** | ViT-S/16 | SGD + Momentum (0.9) | $1 \times 10^{-2}$ | $1 \times 10^{-4}$ | StepLR |
| **S8** | ViT-S/16 | RMSprop (alpha=0.99) | $1 \times 10^{-4}$ | 0.0 | Cosine Annealing |
| **S9** | ViT-S/16 | AdamW (High LR) | $1 \times 10^{-3}$ | $5 \times 10^{-2}$ | Linear Warmup + Cosine |
| **S10** | ViT-S/16 | AdamW (Low LR) | $1 \times 10^{-5}$ | $5 \times 10^{-2}$ | Linear Warmup + Cosine |

### 2. External Clinical Generalization (LungHist700 Cohort)
Evaluation on 691 clinical histological patches across 37 unique patients (adenocarcinoma, squamous cell carcinoma, benign tissue) using **patient-grouped 5-fold cross-validation** to prevent patient-level data leakage.

---

## 📊 Key Findings & Empirical Results

### LC25000 5-Fold Cross-Validation Performance

| Architecture | Scenario | Test Accuracy (Mean ± SD) | Macro F1-Score | Status |
|:---|:---|:---:|:---:|:---:|
| **ResNet50** | AdamW ($10^{-4}$) | **99.93% ± 0.07%** | **0.9993** | Optimal convergence |
| ResNet50 | SGD ($10^{-2}$) | 98.30% ± 0.42% | 0.9830 | Stable |
| ResNet50 | RMSprop ($10^{-4}$) | 99.80% ± 0.12% | 0.9980 | Highly competitive |
| ResNet50 | AdamW ($10^{-3}$) | 99.77% ± 0.15% | 0.9977 | Robust |
| ResNet50 | AdamW ($10^{-5}$) | 99.63% ± 0.20% | 0.9963 | Slow convergence |
| **ViT-S/16** | AdamW ($10^{-4}$) | **99.97% ± 0.04%** | **0.9997** | Ceiling performance |
| ViT-S/16 | SGD ($10^{-2}$) | **80.77% ± 3.45%** | **0.8062** | **Severe degradation (-19.2%)** |
| ViT-S/16 | RMSprop ($10^{-4}$) | 99.87% ± 0.08% | 0.9987 | Stable |
| ViT-S/16 | AdamW ($10^{-3}$) | 99.53% ± 0.28% | 0.9953 | Optimization strain |
| ViT-S/16 | AdamW ($10^{-5}$) | 99.30% ± 0.35% | 0.9930 | Suboptimal |

> **Key Insight:** While ViT-S/16 matches ResNet50 under adaptive optimizers (AdamW/RMSprop > 99.9%), it exhibits severe fragility under plain SGD (falling to 80.77%), whereas ResNet50 retains 98.30% accuracy due to its built-in inductive spatial bias.

### LungHist700 External Cohort Benchmark (Patient-Grouped 5-Fold CV)

| Model Configuration | Pretraining | External Test Acc | Macro F1 | External AUC-ROC | Statistical Significance |
|:---|:---:|:---:|:---:|:---:|:---:|
| ResNet50 (Fine-tuned) | LC25000 Pretrained | **64.25% ± 4.31%** | **0.6120** | **0.8124** | **Baseline (+15.05% vs. direct transfer)** |
| ResNet50 (Standard) | ImageNet-1k | 55.43% ± 5.22% | 0.5218 | 0.7431 | Welch's $t(4) = 5.34, p = 0.006$ |
| ViT-S/16 (Standard) | ImageNet-1k | 53.11% ± 6.10% | 0.4985 | 0.7180 | ViT lag under small cohort ($N=691$) |

> Direct transfer of models trained strictly on LC25000 to LungHist700 yields 49.20% accuracy, demonstrating that near-100% scores on LC25000 are inflated by domain artifacts. Patient-grouped fine-tuning restores ResNet50 to 64.25% (AUC 0.8124).

### Quantitative XAI Faithfulness Audit ($K = 10$ Pixel Degradation Steps)
Sequential pixel perturbation across 150 test patches (50 per class) demonstrates:
- **ResNet50 Grad-CAM**: Attribution Degradation AUC = **0.421 ± 0.034**
- **ViT-S/16 Attention Rollout**: Attribution Degradation AUC = **0.448 ± 0.029**
- **Random Baseline**: Attribution Degradation AUC = **0.435 ± 0.031**
- **Finding:** Neither Grad-CAM nor Attention Rollout significantly outperformed random pixel degradation in quantitative faithfulness, cautioning against uncritical clinical reliance on saliency heatmaps without rigorous validation.

---

## 📁 Repository Structure

```text
resnet50-vit-lung-histopathology-benchmark/
├── configs/                                 # [3] Exact YAML scenario configs
│   ├── resnet50_main.yaml
│   ├── resnet50_sgd.yaml
│   ├── resnet50_rmsprop.yaml
│   ├── resnet50_lr_1e3.yaml
│   ├── resnet50_lr_1e5.yaml
│   ├── vit_s_main.yaml
│   ├── vit_s_sgd.yaml
│   ├── vit_s_rmsprop.yaml
│   ├── vit_s_lr_1e3.yaml
│   └── vit_s_lr_1e5.yaml
│
├── data_splits/                             # [2] Portable split manifests
│   ├── splits_lc25000.csv                   # LC25000 5-fold CV splits
│   ├── splits_lunghist700_patient_kfold.csv # LungHist700 patient-grouped 5-fold splits
│   └── splits_group_aware_audited.csv       # Audited perceptual hashing deduplication
│
├── figures/                                 # Publication-grade figures (300 DPI)
│   ├── comparison_before_after_resnet50.png
│   ├── confusion_matrix_lc25000_to_lunghist700.png
│   ├── patient_kfold_confusion_matrix.png
│   ├── patient_kfold_roc_curves.png
│   ├── side_by_side_confusion_matrices.png
│   ├── side_by_side_roc_comparison.png
│   ├── epoch_computation_comparison.png
│   ├── per_class_metrics_comparison.png
│   ├── training_validation_curves_lc25000.png
│   ├── xai_quantitative_faithfulness_curves.png
│   └── xai_gradcam_lunghist700_gallery.png
│
├── logs_and_metrics/                        # [4 & 5] Complete experimental telemetry
│   ├── lc25000_scenarios/                   # 10 scenario folders with fold histories & metrics
│   │   ├── resnet50_main/ (fold_1..5 history.csv + metrics_summary.json)
│   │   ├── resnet50_sgd/
│   │   ├── resnet50_rmsprop/
│   │   ├── resnet50_lr_1e3/
│   │   ├── resnet50_lr_1e5/
│   │   ├── vit_s_main/
│   │   ├── vit_s_sgd/
│   │   ├── vit_s_rmsprop/
│   │   ├── vit_s_lr_1e3/
│   │   └── vit_s_lr_1e5/
│   ├── lunghist700_external_validation/     # External cohort metrics & OOF predictions
│   └── xai_faithfulness/                    # Quantitative pixel degradation logs
│
├── scripts/                                 # [6] Figure & analysis generation scripts
│   ├── generate_all_scenario_figures.py
│   ├── generate_xai_visual_gallery.py
│   └── phash_grouping.py
│
├── src/                                     # [1] Training and evaluation code
│   ├── dataset.py                           # Dataset loading & augmentation pipelines
│   ├── models.py                            # ResNet50 & ViT-S/16 model factories
│   ├── train.py                             # Main stratified 5-fold training engine
│   ├── evaluate.py                          # Multi-class metrics & confusion matrix evaluator
│   ├── train_eval_grouped_kfold.py          # Patient-grouped cross-validation engine
│   └── evaluate_xai_faithfulness.py         # Quantitative pixel-flipping perturbation engine
│
├── .gitignore
├── .zenodo.json                             # Metadata for permanent Zenodo DOI archiving
├── CITATION.cff                             # Machine-readable academic citation metadata
├── LICENSE                                  # MIT Open Source License
├── README.md                                # Comprehensive replication documentation
└── requirements.txt                         # Python dependencies
```

---

## 🚀 Quickstart & Replication Guide

### 1. Environment Setup

Clone the repository and install required dependencies:
```bash
git clone https://github.com/attaramadhani/resnet50-vit-lung-histopathology-benchmark.git
cd resnet50-vit-lung-histopathology-benchmark
pip install -r requirements.txt
```

### 2. Dataset Preparation
- **LC25000**: Available via Kaggle (`andrewmvd/lung-and-colon-cancer-histopathological-images`) or Academic Torrents. Extract the lung subset containing `lung_aca`, `lung_bma`, and `lung_n`.
- **LungHist700**: Available via Diosdado et al. (Zenodo / Hospital Universitario de Gran Canaria Dr. Negrín).

Place datasets into local directories and verify paths against `data_splits/splits_lc25000.csv` and `data_splits/splits_lunghist700_patient_kfold.csv`.

### 3. Re-running Training Scenarios

To train a specific scenario (e.g., S1: ResNet50 AdamW):
```bash
python src/train.py --config configs/resnet50_main.yaml --data_dir /path/to/lc25000/
```

To run patient-grouped 5-fold cross-validation on LungHist700:
```bash
python src/train_eval_grouped_kfold.py --split_csv data_splits/splits_lunghist700_patient_kfold.csv --data_dir /path/to/lunghist700/
```

### 4. Evaluating Quantitative XAI Faithfulness

To reproduce the 150-sample pixel degradation audit:
```bash
python src/evaluate_xai_faithfulness.py --model_type resnet50 --steps 10 --samples_per_class 50
```

### 5. Regenerating Figures

To reproduce publication figures from experimental logs:
```bash
python scripts/generate_all_scenario_figures.py
```

---

## 📜 Citation

If you use this benchmark, code, or experimental telemetry in your research, please cite our article:

```bibtex
@article{setiawan2026beyond,
  title={Beyond the LC25000 ceiling: optimizer robustness and external validity of a convolutional network and a vision transformer for lung cancer histopathology classification},
  author={Setiawan, Wahyudi and Tri Hida, Attala Alif Ramadhani and Saputra, Ade Verdaus},
  journal={Computer Methods and Programs in Biomedicine Update},
  year={2026},
  publisher={Elsevier},
  note={Official replication package archived on Zenodo}
}
```

---

## 🏆 Funding & Acknowledgments

This research was funded by **Kementerian Pendidikan, Kebudayaan, Riset, dan Teknologi Republik Indonesia (Kemendikbudristek RI)** under the **BIMA 2026 Research Grant** (*Penelitian Terapan Unggulan Perguruan Tinggi*), hosted at **Universitas Trunojoyo Madura**.
