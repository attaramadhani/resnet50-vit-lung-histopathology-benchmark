# Beyond the LC25000 ceiling: optimizer robustness and external validity of a convolutional network and a vision transformer for lung cancer histopathology classification

[![Python](https://img.shields.io/badge/python-3.12.3-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.8.0%20%2B%20CUDA%2012.8-ee4c2c.svg)](https://pytorch.org/)
[![Release](https://img.shields.io/badge/release-v2.1.1-success.svg)](https://github.com/attaramadhani/resnet50-vit-lung-histopathology-benchmark/releases/tag/v2.1.1)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23271474.svg)](https://doi.org/10.5281/zenodo.23271474)
[![Elsevier](https://img.shields.io/badge/Elsevier-CMPB_Update-orange.svg)](https://www.sciencedirect.com/journal/computer-methods-and-programs-in-biomedicine-update)

Official **v2.1.1** open-science replication package (Revision R1 + complete **v2.0.0** baseline) for manuscript **CMPBUP-D-26-00450** submitted to Elsevier's *Computer Methods and Programs in Biomedicine Update* (CMPB Update).

> **Version Note (`v1.0.0` Superseded; `v2.0.0` Preserved):** Release `v1.0.0` (archived at git tag `v1.0.0`) is **superseded and obsolete**. **`v2.0.0` / `v2.0.1`** remain preserved untouched under `logs_and_metrics/` and `figures/`. **`v2.1.1`** adds all Revision R1 experiments, scripts, predictions, and updated figures separately under `results_rev1/`, `figures_rev1/`, and `runpod_rev1/`. See [`CHANGELOG.md`](CHANGELOG.md) for full details.

---

## Script → Manuscript Table & Figure Mapping

| Script | Purpose | Manuscript Output(s) |
| :--- | :--- | :--- |
| [`src/train_scenarios.py`](src/train_scenarios.py) | 14-scenario LC25000 5-fold CV & 3,000-image soft-voting test | **Table 2, Table 3 (rows 1–14), Fig. 2, Fig. 3, Fig. S1** |
| [`src/exp_stats_paired.py`](src/exp_stats_paired.py) | Paired & Nadeau–Bengio corrected resampled *t*-tests | **Section 4.1, Section 4.2** |
| [`src/exp4_duplicate_audit.py`](src/exp4_duplicate_audit.py) | Rotation-invariant dihedral ResNet50 + ORB/RANSAC duplicate audit | **Section 3.2, Section 4.1 (1,779 groups)** |
| [`src/train_clean_grouped.py`](src/train_clean_grouped.py) | Group-aware re-training of S1 & S6 (`G1–G2`) | **Table 3 (rows G1–G2)** |
| [`src/exp3_compute_benchmark.py`](src/exp3_compute_benchmark.py) | GPU-only throughput/latency/VRAM & wall-clock log summary | **Table 4, Fig. 4** |
| [`src/exp2_faithfulness.py`](src/exp2_faithfulness.py) | Grad-CAM & Attention Rollout deletion/insertion AUC (`K=10/20`, 3 baselines) | **Table 5, Fig. 5** |
| [`src/exp1_zeroshot_ensemble.py`](src/exp1_zeroshot_ensemble.py) | Zero-shot 5-fold ensemble test on LungHist700 (`N=691`) | **Table 6 (Zero-shot), Fig. 7a–b** |
| [`src/exp7_lunghist_finetune.py`](src/exp7_lunghist_finetune.py) | Patient-grouped & image-level LungHist700 5-fold CV (inner val, patience 8) | **Table 6 (Fine-tuned & Image-level), Fig. 7c–e, Fig. 8** |
| [`src/lunghist_stats.py`](src/lunghist_stats.py) | Durkalski clustered McNemar + Holm correction on LungHist700 | **Section 4.5** |
| [`src/verify_vit_checkpoint.py`](src/verify_vit_checkpoint.py) | Pretrained weight equivalence (`timm` vs `WinKawaks/vit-small-patch16-224`) | **Section 3.3** |
| [`scripts/generate_manuscript_figures.py`](scripts/generate_manuscript_figures.py) | Reproduces 300 DPI plots from `logs_and_metrics/` | **Figs. 2, 3, 4, 5, 7, 8, Fig. S1** |
| [`scripts/generate_xai_visual_gallery_v2.py`](scripts/generate_xai_visual_gallery_v2.py) | 6 test-image XAI gallery (first 2/class from `exp2_images.csv`) | **Fig. 6** |
| [`scripts/wp01_shortcut_lc25000.py`](scripts/wp01_shortcut_lc25000.py) | LC25000 shallow-feature shortcut test (6 feature sets × 2 classifiers) | **Section 5.4, Revision WP01 ([`results_rev1/WP01/`](results_rev1/WP01/))** |
| [`scripts/wp01b_shortcut_lunghist700.py`](scripts/wp01b_shortcut_lunghist700.py) | LungHist700 negative-control shortcut test (patient-grouped 5-fold CV, `n=691`) | **Section 5.4, Revision WP01b ([`results_rev1/WP01b/`](results_rev1/WP01b/))** |
| [`scripts/wp02_audit_validation.py`](scripts/wp02_audit_validation.py) | Synthetic sibling validation & sensitivity grid for Exp 4 audit | **Revision WP02 ([`results_rev1/WP02/`](results_rev1/WP02/))** |
| [`runpod_rev1/02_run_wp03_wp05.sh`](runpod_rev1/02_run_wp03_wp05.sh) | Matched-recipe (`AdamW`, `wd=0.05`, `warmup=5+cosine`, `freeze=0`) & G1/G2 OOF export | **Revision WP03 & WP05 ([`results_rev1/WP03/`](results_rev1/WP03/), [`WP05/`](results_rev1/WP05/))** |
| [`runpod_rev1/03_run_wp04a_multiseed.sh`](runpod_rev1/03_run_wp04a_multiseed.sh) | Multi-seed replication (`seeds 1, 2`) across S1–S14 & G1–G2 | **Revision WP04a ([`results_rev1/WP04a/`](results_rev1/WP04a/))** |
| [`runpod_rev1/04_run_wp04b_lr_sweep.sh`](runpod_rev1/04_run_wp04b_lr_sweep.sh), [`05_run_wp04c_warmup_cosine.sh`](runpod_rev1/05_run_wp04c_warmup_cosine.sh) | 7-point LR sweep (`seed 42`, SGDM `m=0.9`) & Warmup+Cosine ablation | **Revision WP04b & WP04c ([`results_rev1/WP04b/`](results_rev1/WP04b/), [`WP04c/`](results_rev1/WP04c/))** |
| [`scripts/wp06a_stain_norm_zeroshot.py`](scripts/wp06a_stain_norm_zeroshot.py), [`wp06d`](scripts/wp06d_foundation_linear_probe.py), [`wp06e`](scripts/wp06e_stronger_protocol.py), [`wp06f`](scripts/wp06f_domain_shift.py) | LungHist700 stain norm zero-shot, HED jitter, LC25000-init ViT, Phikon/CTransPath linear probe, tiling 512 + orig 80/10/10, & domain shift | **Revision WP06 ([`results_rev1/WP06/`](results_rev1/WP06/))** |
| [`scripts/wp07_xai_lightweight.py`](scripts/wp07_xai_lightweight.py) | Adebayo cascading randomization (`WP07-c`), spatially-matched Gaussian random baseline (`WP07-a`), per-class median/IQR (`WP07-d`) | **Revision WP07 ([`results_rev1/WP07/`](results_rev1/WP07/))** |
| [`scripts/wp05_wp06_bootstrap_stats.py`](scripts/wp05_wp06_bootstrap_stats.py) | Wilson 95% CI, 2,000× cluster-bootstrap CI, ECE (15-bin), Brier, per-class sens/spec | **[`results_rev1/Statistik_CI_Kalibrasi_SensSpec.csv`](results_rev1/Statistik_CI_Kalibrasi_SensSpec.csv)** |

---

## Authors & Affiliations

- **Wahyudi Setiawan** (Corresponding author: `wsetiawan@trunojoyo.ac.id`, ORCID: [0000-0003-0068-1443](https://orcid.org/0000-0003-0068-1443))
- **Attala Alif Ramadhani Tri Hida**
- **Ade Verdaus Saputra**

*Department of Information Systems, Faculty of Engineering, Universitas Trunojoyo Madura, Bangkalan 69162, East Java, Indonesia*

---

## Repository Structure

> Click any folder or file name below to jump directly to it on GitHub.

- [`configs/`](configs/) — All 14 scenario YAMLs (`S1–S14`) + group-aware `G1–G2` (`16` YAML configs)
- [`data_splits/`](data_splits/) — Deterministic split manifests & SHA-256 verified CSVs
  - [`lc25000_lung_all.csv`](data_splits/lc25000_lung_all.csv) — `15,000` LC25000 lung images
  - [`lc25000_dev.csv`](data_splits/lc25000_dev.csv) — `12,000` development images (`4,000`/class)
  - [`lc25000_test.csv`](data_splits/lc25000_test.csv) — `3,000` held-out test images (`1,000`/class)
  - [`exp4_grouped_split.csv`](data_splits/exp4_grouped_split.csv) — `1,779` rotation-invariant duplicate groups split
  - [`exp4_groups.csv`](data_splits/exp4_groups.csv) & [`exp4_stats.json`](data_splits/exp4_stats.json) — Image-to-group cluster map & summary (Exp 4)
  - [`exp2_images.csv`](data_splits/exp2_images.csv) — `150` test images (`50`/class, seed `42`) for Exp 2 XAI
  - [`lunghist700.csv`](data_splits/lunghist700.csv) — `691` LungHist700 images with patient & magnification IDs
  - [`folds_patient.csv`](data_splits/folds_patient.csv) — 5-fold patient-grouped LungHist700 splits (`136–142` images/fold)
- [`src/`](src/) — Unified PyTorch experimental pipeline (`v2.0.0`, `17` modules)
  - [`common.py`](src/common.py) — ResNet50 & `timm` ViT-S/16 builders, augmentations, metrics, cluster bootstrap
  - [`prepare_cache_256.py`](src/prepare_cache_256.py) — Pre-caches `256×256` LC25000 copies (runtime I/O optimization)
  - [`download_lunghist700.py`](src/download_lunghist700.py) — Automatic LungHist700 fetcher (Figshare DOI)
  - [`verify_vit_checkpoint.py`](src/verify_vit_checkpoint.py) — Weight equivalence test: `timm` augreg vs `WinKawaks/vit-small-patch16-224`
  - [`train_scenarios.py`](src/train_scenarios.py) & [`train_main_folds.py`](src/train_main_folds.py) — 14-scenario 5-fold CV + soft-voting test evaluator
  - [`exp4_duplicate_audit.py`](src/exp4_duplicate_audit.py) — Dihedral ResNet50 embeddings + *k*-NN + ORB/RANSAC + union-find
  - [`train_clean_grouped.py`](src/train_clean_grouped.py) — Group-aware re-training (`G1` & `G2`)
  - [`exp5_train_sgdm.py`](src/exp5_train_sgdm.py) — SGD with momentum (`m=0.9`) training runner
  - [`exp3_compute_benchmark.py`](src/exp3_compute_benchmark.py) — Synthetic GPU-only profiling + wall-clock log summarizer
  - [`exp2_faithfulness.py`](src/exp2_faithfulness.py) — Grad-CAM & Attention Rollout deletion/insertion AUC (`K=10/20`, 3 baselines)
  - [`exp1_zeroshot_ensemble.py`](src/exp1_zeroshot_ensemble.py) — Zero-shot ensemble evaluation on LungHist700
  - [`exp7_lunghist_finetune.py`](src/exp7_lunghist_finetune.py) — Patient-grouped 5-fold CV with inner validation (`~15%` patients, patience `8`)
  - [`exp_stats_paired.py`](src/exp_stats_paired.py) & [`lunghist_stats.py`](src/lunghist_stats.py) — Nadeau–Bengio *t*-tests & Durkalski clustered McNemar + Holm correction
  - [`package_results.py`](src/package_results.py) & [`sync_results_aliases.py`](src/sync_results_aliases.py) — Result aggregation & path alias sync utilities
- [`scripts/`](scripts/) — Publication figure generators & Revision R1 experiments (`12` scripts)
  - [`generate_manuscript_figures.py`](scripts/generate_manuscript_figures.py) — Reproduces Figs. 2–5, 7–8, S1
  - [`generate_xai_visual_gallery_v2.py`](scripts/generate_xai_visual_gallery_v2.py) — Generates updated Fig. 6 XAI visual gallery
  - [`wp01_shortcut_lc25000.py`](scripts/wp01_shortcut_lc25000.py) — Shallow-feature shortcut test on LC25000
  - [`wp01b_shortcut_lunghist700.py`](scripts/wp01b_shortcut_lunghist700.py) — Negative-control shallow-feature shortcut test on LungHist700
  - [`wp02_audit_validation.py`](scripts/wp02_audit_validation.py) — Near-duplicate audit validation & sensitivity grid
  - [`wp05_export_old_g1_g2_preds.py`](scripts/wp05_export_old_g1_g2_preds.py) & [`wp05_wp06_bootstrap_stats.py`](scripts/wp05_wp06_bootstrap_stats.py) — G1/G2 predictions export & cluster-bootstrap/calibration stats
  - [`wp06a_stain_norm_zeroshot.py`](scripts/wp06a_stain_norm_zeroshot.py), [`wp06d_foundation_linear_probe.py`](scripts/wp06d_foundation_linear_probe.py), [`wp06e_stronger_protocol.py`](scripts/wp06e_stronger_protocol.py), [`wp06f_domain_shift.py`](scripts/wp06f_domain_shift.py) — LungHist700 robustness, foundation probes, & domain shift
  - [`wp07_xai_lightweight.py`](scripts/wp07_xai_lightweight.py) — Lightweight XAI sanity checks (Adebayo randomization, matched random, median/IQR)
- [`figures/`](figures/) — Original `v2.0.0/v2.0.1` manuscript figures (`Figs 2–8` & `Fig. S1`, preserved untouched)
- [`figures_rev1/`](figures_rev1/) — Revision R1 updated figures (`Fig. 2, 6, 7, 8` fixes + `WP04b` LR sweep & `WP06` plots)
- [`results_rev1/`](results_rev1/) — Complete Revision R1 logs, configs, metrics, & per-image CSV predictions
  - [`WP01/`](results_rev1/WP01/) — LC25000 shallow color/thumbnail shortcut tests
  - [`WP01b/`](results_rev1/WP01b/) — LungHist700 negative-control shortcut test ([`wp01b_results.md`](results_rev1/WP01b/wp01b_results.md), [`wp01b_results.csv`](results_rev1/WP01b/wp01b_results.csv), [`wp01b_terminal.log`](results_rev1/WP01b/wp01b_terminal.log))
  - [`WP02/`](results_rev1/WP02/) — Near-duplicate audit grid (`64` configs, `150` synthetic families)
  - [`WP03/`](results_rev1/WP03/) — Matched-recipe 3-seed comparison (image-level vs group-aware, `72` pred CSVs)
  - [`WP04a/`](results_rev1/WP04a/), [`WP04b/`](results_rev1/WP04b/), [`WP04c/`](results_rev1/WP04c/) — Multi-seed `S1–S14/G1–G2`, 7-point LR sweep (SGDM `m=0.9`), Warmup+Cosine
  - [`WP05/`](results_rev1/WP05/) — Clean baseline `G1/G2` 5-fold + ensemble predictions ([`exp4_grouped_split.csv`](data_splits/exp4_grouped_split.csv))
  - [`WP06/`](results_rev1/WP06/) — LungHist700: `wp06a–wp06f` (Phikon, CTransPath, tiling, 80/10/10, FID/MMD)
  - [`WP07/`](results_rev1/WP07/) — Lightweight XAI: Adebayo randomization, spatial-matched random, median/IQR
  - [`COMPUTE/`](results_rev1/COMPUTE/) — Single-session GPU compute profiling (ResNet50, ViT frozen, ViT unfrozen)
  - [`FIG6/`](results_rev1/FIG6/) — Updated `Fig. 6` (`ViT-S/16 rollout map`)
  - [`logs/`](results_rev1/logs/) — Complete terminal execution logs
  - [`Statistik_CI_Kalibrasi_SensSpec.csv`](results_rev1/Statistik_CI_Kalibrasi_SensSpec.csv) — Cluster-bootstrap CIs, ECE, Brier, per-class sens/spec
  - [`COMPLETENESS_AUDIT.txt`](results_rev1/COMPLETENESS_AUDIT.txt), [`MANIFEST_SHA256.txt`](results_rev1/MANIFEST_SHA256.txt), [`environment_rev1.txt`](results_rev1/environment_rev1.txt), [`scripts_sha256_rev1.txt`](results_rev1/scripts_sha256_rev1.txt)
- [`runpod_rev1/`](runpod_rev1/) — Reproducible Revision R1 orchestration & verification scripts ([`RUNPOD_SETUP_AND_SYNC_GUIDE.md`](runpod_rev1/RUNPOD_SETUP_AND_SYNC_GUIDE.md), [`run_master_rev1.sh`](runpod_rev1/run_master_rev1.sh))
- [`logs_and_metrics/`](logs_and_metrics/) — Raw experimental logs & compiled tables (`v2.0.0/v2.0.1`)
  - [`tables_for_manuscript.md`](logs_and_metrics/tables_for_manuscript.md) — Auto-compiled Tables 2–6 with file provenance
  - [`environment.txt`](logs_and_metrics/environment.txt) — Software versions, NVIDIA RTX 4090 info, and SHA-256 checksums
  - [`vit_checkpoint_check.json`](logs_and_metrics/vit_checkpoint_check.json) — Verified `max_abs_diff = 2.24e-5`, `top1_agreement = 1.0`
  - [`run_all.log`](logs_and_metrics/run_all.log) — Complete RunPod execution log (`5,706` lines)
  - [`runs/`](logs_and_metrics/runs/) — Per-scenario `config.yaml`, fold histories, and `metrics.json` (`S1–S14`, `G1–G2`)
  - [`exp2_faithfulness/`](logs_and_metrics/exp2_faithfulness/) — Per-image & summary CSV/TXT/JSON (`K=10/20`, `grey`/`blur`/`tissue_mean`, logits)
  - [`exp4/`](logs_and_metrics/exp4/) — Duplicate audit pairs, group stats, and clean benchmark results
  - [`lunghist700/`](logs_and_metrics/lunghist700/) — OOF predictions, fold logs, metrics JSONs, and clustered McNemar tests
- [`run_all.sh`](run_all.sh) — Master shell script executing the full pipeline end-to-end
- [`requirements.txt`](requirements.txt) — Exact package versions (§3.8)
- [`CHANGELOG.md`](CHANGELOG.md) | [`CITATION.cff`](CITATION.cff) | [`.zenodo.json`](.zenodo.json) | [`LICENSE`](LICENSE)

---

## Key Protocols & Architectural Specifications (v2.0.0)

1. **Models (§3.3)**:
   - **ResNet50**: `torchvision` `IMAGENET1K_V2` with `Dropout(0.3) + Linear(2048, 3)` (`23.51M` parameters).
   - **ViT-S/16**: `timm` `vit_small_patch16_224.augreg_in21k_ft_in1k` without dropout (`0.0`), encoder blocks `0` and `1` frozen (`21.67M` total / `18.12M` trainable parameters). Tensor equivalence against HuggingFace `WinKawaks/vit-small-patch16-224` was verified to within `2.24 × 10⁻⁵` (`top1_agreement = 1.0`, see `logs_and_metrics/vit_checkpoint_check.json`).
   - Both architectures receive `224 × 224` RGB inputs normalized with ImageNet statistics.
2. **Training Protocol (§3.4)**:
   - Mixed precision FP16, batch size `32`, cross-entropy loss, weight decay `1e-4`, **no LR scheduler**, **no gradient clipping**, max `50` epochs, early stopping patience `5` on `val_loss` (`min_delta = 0.0`), seed `42` reset at the start of every fold.
   - On-the-fly training augmentations: horizontal flip (`p=0.5`), random rotation (`±20°`), random affine (`scale=0.9–1.1`, `shear=±10°`), and color jitter (`brightness=0.2`, `contrast=0.2`). *(LC25000 raw `768×768` images were cached once at `256×256` px in `data/` via `src/prepare_cache_256.py` purely to eliminate disk I/O overhead; `224×224` resizing and augmentations occurred on the fly.)*
3. **Near-Duplicate Audit (§3.2, Exp 4)**:
   - Rotation-invariant audit via 8-dihedral averaged ImageNet-pretrained ResNet50 embeddings, class-wise top-30 cosine nearest neighbors (`cos >= 0.90`), geometric verification with `ORB` + `RANSAC` (`>= 25` inliers), and union–find clustering (`1,779` groups) followed by `StratifiedGroupKFold` re-training (`G1–G2`).
4. **XAI Faithfulness (§3.6, Exp 2)**:
   - Evaluates Grad-CAM (`layer4`, raw logit) and Attention Rollout (residual identity across 12 blocks) on 150 balanced test patches (`data_splits/exp2_images.csv`, seed `42`), using `K=10` (primary) and `K=20` steps under `grey`, `blur`, and `tissue_mean` baselines, compared against **separate per-model random attribution baselines** via paired two-sided Wilcoxon signed-rank tests.
5. **External Validation on LungHist700 (§3.7, Exp 1, 6, 7)**:
   - Zero-shot evaluation of the 5-fold S1 & S6 ensembles across all `691` images (`45` patients).
   - Patient-grouped 5-fold fine-tuning (`data_splits/folds_patient.csv`, 136–142 images per fold) using **inner patient-grouped validation (~15% patients)** for early stopping (`patience=8` on `val_loss`, no LR scheduler), ensuring held-out test folds never inform training or checkpoint selection. Includes image-level split ablation (`Exp 6`) and per-magnification models (`20×`, `40×`), with 95% patient-level cluster bootstrap CIs and Durkalski clustered McNemar tests.

---

## Summary of Main Empirical Results

### LC25000 14 Scenarios + Group-Aware Re-runs (Table 3)

| No. | Scenario | Model | Optimizer & LR | 5-Fold CV Acc (%) | Held-out Test Acc (%) | Macro F1 |
| :---: | :--- | :--- | :--- | :---: | :---: | :---: |
| 1 | `resnet50_main` | ResNet50 | Adam `1e-4` | `99.99 ± 0.02` | `100.00` [99.9–100.0] | `1.0000` |
| 2 | `resnet50_rmsprop` | ResNet50 | RMSprop `1e-4` | `100.00 ± 0.00` | `100.00` [99.9–100.0] | `1.0000` |
| 3 | `resnet50_lr_1e3` | ResNet50 | Adam `1e-3` | `98.62 ± 0.76` | `99.50` [99.2–99.7] | `0.9950` |
| 4 | `resnet50_lr_1e5` | ResNet50 | Adam `1e-5` | `99.99 ± 0.02` | `100.00` [99.9–100.0] | `1.0000` |
| 5 | `resnet50_sgd` | ResNet50 | SGD (`m=0.0`) `1e-4` | `94.35 ± 0.41` | `95.00` [94.2–95.7] | `0.9500` |
| 6 | `vit_s_main` | ViT-S/16 | AdamW `1e-4` | `99.99 ± 0.02` | `100.00` [99.9–100.0] | `1.0000` |
| 7 | `vit_s_rmsprop` | ViT-S/16 | RMSprop `1e-4` | `99.97 ± 0.03` | `100.00` [99.9–100.0] | `1.0000` |
| 8 | `vit_s_lr_1e3` | ViT-S/16 | AdamW `1e-3` | `94.79 ± 1.45` | `96.27` [95.5–96.9] | `0.9627` |
| 9 | `vit_s_lr_1e5` | ViT-S/16 | AdamW `1e-5` | `100.00 ± 0.00` | `100.00` [99.9–100.0] | `1.0000` |
| 10 | `vit_s_sgd` | ViT-S/16 | SGD (`m=0.0`) `1e-4` | `98.79 ± 0.27` | `98.83` [98.4–99.2] | `0.9883` |
| 11 | `resnet50_sgdm` | ResNet50 | SGD (`m=0.9`) `1e-4` | `99.37 ± 0.26` | `99.30` [98.9–99.5] | `0.9930` |
| 12 | `vit_s_sgdm` | ViT-S/16 | SGD (`m=0.9`) `1e-4` | `99.93 ± 0.06` | `99.97` [99.8–100.0] | `0.9997` |
| 13 | `resnet50_sgdm_lr_1e3` | ResNet50 | SGD (`m=0.9`) `1e-3` | `99.99 ± 0.02` | `100.00` [99.9–100.0] | `1.0000` |
| 14 | `vit_s_sgdm_lr_1e3` | ViT-S/16 | SGD (`m=0.9`) `1e-3` | `100.00 ± 0.00` | `100.00` [99.9–100.0] | `1.0000` |
| G1 | `clean_resnet50` | ResNet50 | Adam `1e-4` (group-aware) | `98.93 ± 0.60` | `99.97` [99.8–100.0] | `0.9997` |
| G2 | `clean_vit_s16` | ViT-S/16 | AdamW `1e-4` (group-aware) | `99.14 ± 0.43` | `99.93` [99.8–100.0] | `0.9993` |

### LungHist700 External Validation (Table 6, N = 691 Images, 45 Patients)

| Setting | Model | OOF Accuracy (%) [Cluster 95% CI] | Balanced Acc (%) | Macro F1 | Macro ROC-AUC |
| :--- | :--- | :---: | :---: | :---: | :---: |
| Zero-shot | ResNet50 (S1 ensemble) | `53.11` [41.4–65.1] | `48.83` | `0.4661` | `0.7145` |
| Zero-shot | ViT-S/16 (S6 ensemble) | `54.41` [43.2–66.2] | `49.92` | `0.4830` | `0.7422` |
| Patient-grouped FT | ResNet50 (ImageNet) | `68.89` [62.7–75.2] | `70.92` | `0.7073` | `0.8511` |
| Patient-grouped FT | ResNet50 (LC25000 init) | `70.77` [64.3–76.8] | `73.31` | `0.7212` | `0.8585` |
| Patient-grouped FT | ViT-S/16 (ImageNet) | `69.46` [62.6–76.3] | `72.16` | `0.7163` | `0.8621` |
| Image-level ablation | ResNet50 (ImageNet) | `84.37` [80.8–88.2] | `85.36` | `0.8483` | `0.9521` |
| Image-level ablation | ViT-S/16 (ImageNet) | `85.09` [80.7–88.9] | `85.91` | `0.8548` | `0.9557` |

---

## Reproducing Experiments & Figures

```bash
git clone https://github.com/attaramadhani/resnet50-vit-lung-histopathology-benchmark.git
cd resnet50-vit-lung-histopathology-benchmark
pip install -r requirements.txt

# 1. Reproduce updated manuscript figures (Figs 2-5, 7-8, S1) into figures_rev1/:
python scripts/generate_manuscript_figures.py --out-dir figures_rev1

# 2. Recompute bootstrap CIs, calibration (ECE/Brier), and per-class sensitivity/specificity from results_rev1/:
python scripts/wp05_wp06_bootstrap_stats.py --results-root results_rev1 --out results_rev1/Statistik_CI_Kalibrasi_SensSpec.csv

# 3. Run the WP01b negative-control shortcut test on LungHist700 (CPU):
python scripts/wp01b_shortcut_lunghist700.py --root /path/to/LungHist700 --out results_rev1/WP01b

# 4. Run the v2.0.0 baseline or full Revision R1 Paket C pipeline on GPU (requires Kaggle/Figshare datasets):
bash run_all.sh
bash runpod_rev1/run_master_rev1.sh --paket C
```

---

## Citation

```bibtex
@article{setiawan2026beyond,
  title={Beyond the LC25000 ceiling: optimizer robustness and external validity of a convolutional network and a vision transformer for lung cancer histopathology classification},
  author={Setiawan, Wahyudi and Tri Hida, Attala Alif Ramadhani and Saputra, Ade Verdaus},
  journal={Computer Methods and Programs in Biomedicine Update},
  year={2026},
  publisher={Elsevier},
  note={Replication package archived at Zenodo (Concept DOI): https://doi.org/10.5281/zenodo.23271474}
}
```

## Funding

Funded by the **Ministry of Higher Education, Science, and Technology, Republic of Indonesia**, through the **BIMA** research grant scheme *Penelitian Fundamental Reguler*, no. `B/036/UN46.1/PT.01.03/BIMA/PL-MULTITAHUNLANJUTAN/2026`.
