# Manuscript Tables Summary — CMPB Update Revision

Generated automatically by `package_results.py` directly from raw experimental logs.

Date generated: 2026-10-15 06:47:37

---

## Table 2 & 3: LC25000 5-Fold Cross-Validation and 3,000-Image Ensemble Test

Protocol: 12,000 dev images (4,000/class, 5-fold CV) + 3,000 test images (1,000/class soft voting).

| No. | Scenario Name | Architecture | Optimizer | LR | Status | 5-Fold CV Val Acc (%) | Ensemble Test Acc [95% CI] | Macro F1 | Macro AUC | Mean Stop Ep | Source Log |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | `resnet50_main` | ResNet50 | Adam | 1e-4 | Draft (re-run) | 99.99 ± 0.02% | 100.00% [99.87-100.00%] (3000/3000) | 1.0000 | 1.0000 | 17.2 | `runs/resnet50_main/metrics.json` |
| 2 | `resnet50_rmsprop` | ResNet50 | RMSprop | 1e-4 | Draft (re-run) | 100.00 ± 0.00% | 100.00% [99.87-100.00%] (3000/3000) | 1.0000 | 1.0000 | 15.0 | `runs/resnet50_rmsprop/metrics.json` |
| 3 | `resnet50_lr_1e3` | ResNet50 | Adam | 1e-3 | Draft (re-run) | 98.62 ± 0.76% | 99.50% [99.18-99.70%] (2985/3000) | 0.9950 | 0.9999 | 14.0 | `runs/resnet50_lr_1e3/metrics.json` |
| 4 | `resnet50_lr_1e5` | ResNet50 | Adam | 1e-5 | Draft (re-run) | 99.99 ± 0.02% | 100.00% [99.87-100.00%] (3000/3000) | 1.0000 | 1.0000 | 30.8 | `runs/resnet50_lr_1e5/metrics.json` |
| 5 | `resnet50_sgd` | ResNet50 | SGD m=0 | 1e-4 | Draft (re-run) | 94.35 ± 0.41% | 95.00% [94.16-95.72%] (2850/3000) | 0.9500 | 0.9935 | 50.0 | `runs/resnet50_sgd/metrics.json` |
| 6 | `vit_s_main` | ViT-S/16 | AdamW | 1e-4 | Draft (re-run) | 99.99 ± 0.02% | 100.00% [99.87-100.00%] (3000/3000) | 1.0000 | 1.0000 | 14.4 | `runs/vit_s_main/metrics.json` |
| 7 | `vit_s_rmsprop` | ViT-S/16 | RMSprop | 1e-4 | Draft (re-run) | 99.97 ± 0.03% | 100.00% [99.87-100.00%] (3000/3000) | 1.0000 | 1.0000 | 21.4 | `runs/vit_s_rmsprop/metrics.json` |
| 8 | `vit_s_lr_1e3` | ViT-S/16 | AdamW | 1e-3 | Draft (re-run) | 94.79 ± 1.45% | 96.27% [95.53-96.89%] (2888/3000) | 0.9627 | 0.9965 | 19.6 | `runs/vit_s_lr_1e3/metrics.json` |
| 9 | `vit_s_lr_1e5` | ViT-S/16 | AdamW | 1e-5 | Draft (re-run) | 100.00 ± 0.00% | 100.00% [99.87-100.00%] (3000/3000) | 1.0000 | 1.0000 | 20.2 | `runs/vit_s_lr_1e5/metrics.json` |
| 10 | `vit_s_sgd` | ViT-S/16 | SGD m=0 | 1e-4 | Draft (re-run) | 98.79 ± 0.27% | 98.83% [98.38-99.16%] (2965/3000) | 0.9883 | 0.9997 | 50.0 | `runs/vit_s_sgd/metrics.json` |
| 11 | `resnet50_sgdm` | ResNet50 | SGD m=0.9 | 1e-4 | **Baru (Exp 5)** | 99.37 ± 0.26% | 99.30% [98.93-99.54%] (2979/3000) | 0.9930 | 0.9999 | 47.4 | `runs/resnet50_sgdm/metrics.json` |
| 12 | `vit_s_sgdm` | ViT-S/16 | SGD m=0.9 | 1e-4 | **Baru (Exp 5)** | 99.93 ± 0.06% | 99.97% [99.81-99.99%] (2999/3000) | 0.9997 | 1.0000 | 33.6 | `runs/vit_s_sgdm/metrics.json` |
| 13 | `resnet50_sgdm_lr_1e3` | ResNet50 | SGD m=0.9 | 1e-3 | **Baru (Exp 5)** | 99.99 ± 0.02% | 100.00% [99.87-100.00%] (3000/3000) | 1.0000 | 1.0000 | 34.6 | `runs/resnet50_sgdm_lr_1e3/metrics.json` |
| 14 | `vit_s_sgdm_lr_1e3` | ViT-S/16 | SGD m=0.9 | 1e-3 | **Baru (Exp 5)** | 100.00 ± 0.00% | 100.00% [99.87-100.00%] (3000/3000) | 1.0000 | 1.0000 | 26.4 | `runs/vit_s_sgdm_lr_1e3/metrics.json` |
| 4.r | `clean_resnet50` | ResNet50 Group-Aware | Adam/AdamW | 1e-4 | **Exp 4 Group-Aware** | 98.93 ± 0.60% | 99.97% [99.81-99.99%] | 0.9997 | 1.0000 | - | `exp4_clean_benchmark.json` |
| 4.v | `clean_vit_s16` | ViT-S/16 Group-Aware | Adam/AdamW | 1e-4 | **Exp 4 Group-Aware** | 99.14 ± 0.43% | 99.93% [99.76-99.98%] | 0.9993 | 1.0000 | - | `exp4_clean_benchmark.json` |


### Statistical Significance: Paired and Corrected Resampled t-tests (Section 4)

Nadeau-Bengio corrected t-test accounts for 5-fold cross-validation dependency with ratio $r = n_{\text{val}} / n_{\text{train}} = 2400 / 9600 = 0.25$.

| Comparison | Scenario A | Scenario B | Mean Diff (%) | Paired t(4) | Paired p | Corrected t(4) | Corrected p (Nadeau-Bengio) | Source |
|---|---|---|---|---|---|---|---|---|
| Scenario 1 vs 6 (Adam 1e-4 vs AdamW 1e-4 (Primary)) | `resnet50_main` (99.99%) | `vit_s_main` (99.99%) | +0.00% | 0.000 | 1 | 0.000 | 1 | `paired_stats.json` |
| Scenario 5 vs 10 (SGD m=0 lr=1e-4) | `resnet50_sgd` (94.35%) | `vit_s_sgd` (98.79%) | -4.44% | -17.257 | 7e-05 | -11.504 | 0.00033 | `paired_stats.json` |
| Scenario 11 vs 12 (SGD m=0.9 lr=1e-4 (Exp 5)) | `resnet50_sgdm` (99.37%) | `vit_s_sgdm` (99.93%) | -0.57% | -4.732 | 0.00909 | -3.155 | 0.03435 | `paired_stats.json` |
| Scenario 13 vs 14 (SGD m=0.9 lr=1e-3 (Exp 5)) | `resnet50_sgdm_lr_1e3` (99.99%) | `vit_s_sgdm_lr_1e3` (100.00%) | -0.01% | -1.000 | 0.3739 | -0.667 | 0.5415 | `paired_stats.json` |


## Table 4: Computational Profiling on NVIDIA RTX 4090 (Exp 3)

| Architecture | Parameters (M) | MACs (G) | Training Throughput (img/s) | Inference Latency bs=32 (ms/img) | Inference Latency bs=1 (ms/img) | Peak VRAM Train (GB) | Source |
|---|---|---|---|---|---|---|---|
| RESNET50 | 23.51 | 4.11 | 1252.7 ± 13.3 | 0.18 | 5.035 | 1.81 | `exp3_resnet50.json` |
| VIT_S16 | 21.67 | 4.25 | 1635.2 ± 92.7 | 0.149 | 4.898 | 1.4 | `exp3_vit_s16.json` |


**Real-world Training Time**: Total 70 runs duration = **6.4 hours** (23042.6 seconds across 80 logged runs). Mean epoch time = 11.06 s. [Source: `real_training_time_summary.json`]


## Table 5: XAI Faithfulness Evaluation on 150 Test Images (Exp 2)

Evaluates Grad-CAM (ResNet50) and Attention Rollout (ViT-S/16) on 150 test images (`exp2_images.csv`, seed 42).

| Architecture | Checkpoint SHA256 | Attribution Method | Baseline | Deletion AUC (K=10) | Insertion AUC (K=10) | Random Deletion AUC | Random Insertion AUC | Wilcoxon p (method vs random) | Source |
|---|---|---|---|---|---|---|---|---|---|
| RESNET50 | `f9167bfa83146b85...` | Grad-CAM layer4 | grey | 0.6973 ± 0.1583 | 0.8634 ± 0.1469 | 0.5085 ± 0.3582 | 0.5092 ± 0.3581 | del p = 2.3e-06; ins p = 1.01e-15 | `exp2_resnet50_per_image.csv` |
| RESNET50 | `f9167bfa83146b85...` | Grad-CAM layer4 | blur | 0.8954 ± 0.1574 | 0.9296 ± 0.1246 | 0.8487 ± 0.2586 | 0.8500 ± 0.2574 | del p = 0.722; ins p = 0.0118 | `exp2_resnet50_per_image.csv` |
| RESNET50 | `f9167bfa83146b85...` | Grad-CAM layer4 | tissue_mean | 0.7670 ± 0.1414 | 0.9266 ± 0.0784 | 0.5613 ± 0.3355 | 0.5563 ± 0.3394 | del p = 6.95e-12; ins p = 3.54e-16 | `exp2_resnet50_per_image.csv` |
| VIT | `fa8bfad8e8231161...` | Attention Rollout | grey | 0.7447 ± 0.2446 | 0.7585 ± 0.2400 | 0.8107 ± 0.1950 | 0.8078 ± 0.1932 | del p = 0.482; ins p = 0.758 | `exp2_vit_per_image.csv` |
| VIT | `fa8bfad8e8231161...` | Attention Rollout | blur | 0.9537 ± 0.1314 | 0.9600 ± 0.1116 | 0.9557 ± 0.1452 | 0.9564 ± 0.1365 | del p = 3.55e-07; ins p = 4.54e-06 | `exp2_vit_per_image.csv` |
| VIT | `fa8bfad8e8231161...` | Attention Rollout | tissue_mean | 0.8518 ± 0.1777 | 0.8926 ± 0.1290 | 0.7744 ± 0.2174 | 0.7754 ± 0.2155 | del p = 3.92e-11; ins p = 7.05e-16 | `exp2_vit_per_image.csv` |


## Table 6: External Validation on LungHist700 Dataset (691 Images, 45 Patients)

| Experiment | Model & Protocol | Split Mode | Accuracy (%) | Patient Cluster 95% CI | Macro F1 | Macro AUC | Acc 20x / 40x | Source File |
|---|---|---|---|---|---|---|---|---|
| Exp 6 (Image Ablation) | `e6_r50_image` | Image-Level | 84.37% | [80.84-88.19%] | 0.8483 | 0.9521 | 84.68% / 84.04% | `e6_r50_image_metrics.json` |
| Exp 6 (Image Ablation) | `e6_vit_image` | Image-Level | 85.09% | [80.69-88.90%] | 0.8548 | 0.9557 | 86.35% / 83.73% | `e6_vit_image_metrics.json` |
| Exp 7 (Patient-Grouped) | `e7_r50_20x` | Patient-Grouped | 65.74% | [54.49-76.73%] | 0.6616 | 0.8316 | 65.74% / -% | `e7_r50_20x_metrics.json` |
| Exp 7 (Patient-Grouped) | `e7_r50_40x` | Patient-Grouped | 63.86% | [54.97-72.34%] | 0.6566 | 0.8082 | -% / 63.86% | `e7_r50_40x_metrics.json` |
| Exp 7 (Patient-Grouped) | `e7_r50_lcinit` | Patient-Grouped | 70.77% | [64.32-76.77%] | 0.7212 | 0.8585 | 66.85% / 75.0% | `e7_r50_lcinit_metrics.json` |
| Exp 7 (Patient-Grouped) | `e7_r50` | Patient-Grouped | 68.89% | [62.74-75.17%] | 0.7073 | 0.8511 | 68.25% / 69.58% | `e7_r50_metrics.json` |
| Exp 7 (Patient-Grouped) | `e7_vit_20x` | Patient-Grouped | 54.32% | [40.33-68.44%] | 0.5541 | 0.7718 | 54.32% / -% | `e7_vit_20x_metrics.json` |
| Exp 7 (Patient-Grouped) | `e7_vit_40x` | Patient-Grouped | 66.87% | [57.71-75.09%] | 0.6900 | 0.8212 | -% / 66.87% | `e7_vit_40x_metrics.json` |
| Exp 7 (Patient-Grouped) | `e7_vit` | Patient-Grouped | 69.46% | [62.58-76.26%] | 0.7163 | 0.8621 | 68.8% / 70.18% | `e7_vit_metrics.json` |
| Exp 1 (Zero-Shot) | `exp1_resnet50_crop768` | Zero-Shot | 57.45% | [47.62-67.73%] | 0.5232 | 0.7545 | 50.974930362116986% / 64.45783132530121% | `exp1_resnet50_crop768_metrics.json` |
| Exp 1 (Zero-Shot) | `exp1_resnet50_main` | Zero-Shot | 53.11% | [41.42-65.09%] | 0.4661 | 0.7145 | 44.28969359331476% / 62.65060240963856% | `exp1_resnet50_main_metrics.json` |
| Exp 1 (Zero-Shot) | `exp1_vit_crop768` | Zero-Shot | 62.37% | [53.91-71.53%] | 0.5909 | 0.7917 | 58.495821727019504% / 66.56626506024097% | `exp1_vit_crop768_metrics.json` |
| Exp 1 (Zero-Shot) | `exp1_vit_main` | Zero-Shot | 54.41% | [43.16-66.15%] | 0.4830 | 0.7422 | 48.74651810584958% / 60.54216867469879% | `exp1_vit_main_metrics.json` |


### Clustered McNemar Test (Durkalski et al., 2003) on LungHist700

| Model Comparison | Discordant Pairs (n10 / n01) | Clusters (Patients) | Durkalski Chi-Square(1) | Raw p-value | Holm-Adjusted p-value | Source |
|---|---|---|---|---|---|---|
| ResNet50 (ImageNet) vs ViT-S/16 (ImageNet) | n10=86, n01=90 | 45 | 0.0332 | 8.5543e-01 | **1.0000e+00** | `lunghist_statistical_tests.json` |
| ResNet50 (ImageNet) vs ResNet50 (LC25000 init) | n10=74, n01=87 | 45 | 0.3514 | 5.5335e-01 | **1.0000e+00** | `lunghist_statistical_tests.json` |
| ResNet50 (LC25000 init) vs ViT-S/16 (ImageNet) | n10=77, n01=68 | 45 | 0.3011 | 5.8318e-01 | **1.0000e+00** | `lunghist_statistical_tests.json` |

