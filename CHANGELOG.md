# Changelog

## [2.1.0] - 2026-11-24 (Final Revision R1 Release — Preserves `v2.0.0` / `v2.0.1` Baselines)

All Revision R1 artifacts are added under `results_rev1/`, `figures_rev1/`, and `runpod_rev1/` **without overwriting** existing `v2.0.0`/`v2.0.1` `logs_and_metrics/` or `figures/`:
- **WP01 (`results_rev1/WP01/`)**: LC25000 shallow-feature shortcut validation (`colour_hist` RF `97.83%` group-aware / `100.00%` image-level).
- **WP02 (`results_rev1/WP02/`)**: 1,500-image synthetic near-duplicate audit validation (`48.62%` pair recall, `89.50%` precision at default `0.90/30/25`, 64-config grid).
- **WP03 (`results_rev1/WP03/`)**: Matched-recipe (`AdamW`, `lr=1e-4`, `wd=0.05`, `warmup=5+cosine`, `freeze_vit=0`, `head_dropout=0.0`) 3-seed image-level vs group-aware comparison (`60` folds, `72` pred CSVs).
- **WP04a–c (`results_rev1/WP04a..c/`)**: Multi-seed replication (seeds 1 & 2, 160 folds), 7-point LR sweep (seed 42, SGDM `mom=0.9`, `nesterov=False`, 140 folds), and Warmup+Cosine ablation (seeds 42/1/2, 60 folds).
- **WP05 (`results_rev1/WP05/`)**: Per-image predictions for `G1_clean_resnet50` (`99.97%`) and `G2_clean_vit_s16` (`99.93%`) on `exp4_grouped_split.csv`.
- **WP06 (`results_rev1/WP06/`)**: LungHist700 stain normalization zero-shot (`WP06a`), HED jitter & LC25000-init fine-tuning (`WP06b/c`), Phikon (`86.69%`) & CTransPath (`86.11%`) foundation linear probes (`WP06d`), 512×512 tiling & original 80/10/10 (`WP06e`), and domain shift (`FID=76.691`, `MMD^2=0.1633`, `WP06f`).
- **WP07 Lightweight & Single-Session Compute (`results_rev1/WP07/`, `results_rev1/COMPUTE/`)**: Adebayo cascading randomization (`WP07-c`), spatially-matched Gaussian random baseline (`WP07-a`), per-class median/IQR (`WP07-d`), and identical-session GPU profiling (`ResNet50`, `ViT-S/16 frozen`, `ViT-S/16 unfrozen` on Tesla T4).
- **Figures & Statistics (`figures_rev1/`, `results_rev1/Statistik_CI_Kalibrasi_SensSpec.csv`)**: Corrected Fig. 6 (`ViT-S/16 rollout map`), updated Fig. 2/7/8, Rev1 plots, and 2,000× cluster-bootstrap CIs, ECE, Brier, and per-class sensitivity/specificity via `scripts/wp05_wp06_bootstrap_stats.py`.

## [2.0.1] - 2026-11-22 (Hotfix & Revision R1 Tooling)

### Fixed & Added
- **Fixed Missing PIL Import in `src/common.py`**: Added `from PIL import Image` so `ImageList.__getitem__` runs cleanly in standalone replication tests.
- **Revised Fig. 6 (`figures/fig6_xai_multiclass_gallery.png`, `scripts/generate_xai_visual_gallery_v2.py`)**: Regenerated strictly from the first two test images per class in `data_splits/exp2_images.csv` (`lungaca2935`, `lungaca2984`, `lungn2089`, `lungn1327`, `lungscc4868`, `lungscc2554`) without figure title or unvalidated interpretive labels, and removed the superseded BIMA-titled gallery script.
- **Included Revision Verification Scripts (`scripts/`)**: Added `scripts/wp01_shortcut_lc25000.py` (LC25000 shallow-feature shortcut test) and `scripts/wp02_audit_validation.py` (near-duplicate audit sensitivity validation).
- **Updated Documentation (`README.md`)**: Added explicit deprecation notice for `v1.0.0` and a complete Script-to-Manuscript Table/Figure mapping table.

## [2.0.0] - 2026-10-15 (Full Revision Package Aligned with CMPBUP-D-26-00450; Zenodo DOI: `10.5281/zenodo.24432592`)

Replaces the initial `v1.0.0` draft artifacts with the unified revision pipeline executed on NVIDIA RTX 4090 (RunPod, Oct 14–15, 2026) and resolves all code–log–manuscript discrepancies:

### Deprecation Notice for `v1.0.0` (Superseded)
All `v1.0.0` artifacts (`10.5281/zenodo.23271475`) are **superseded** and preserved strictly at git tag `v1.0.0`. Specifically, `v1.0.0` is obsolete because: (1) its YAML configs specified `momentum: 0.9` for SGD and `use_augmentation: false`, (2) `src/train.py` used a `ReduceLROnPlateau` LR scheduler, (3) `src/models.py` loaded ViT-S/16 via HuggingFace with `0.1` dropout, (4) near-duplicate auditing relied only on non-rotation-invariant `phash_grouping.py`, and (5) `src/train_eval_grouped_kfold.py` trained for 4 fixed epochs while stepping its scheduler and selecting checkpoints on the held-out fold without an inner patient-grouped validation split. `v2.0.0`+ replaces every one of those components cleanly.

### Added & Updated
- **Unified 14-Scenario LC25000 Training Engine (`src/train_scenarios.py`, `src/common.py`)**:
  - Covers Scenarios **S1–S14** (including **S11–S14** SGD with momentum `0.9` at `1e-4` and `1e-3`) plus group-aware re-runs **G1–G2** (`src/train_clean_grouped.py`), and all **16 YAML configs** (`configs/*.yaml`).
  - **S5 (`resnet50_sgd`) & S10 (`vit_s_sgd`)** strictly use plain SGD (`momentum = 0.0`) for full 50 epochs (ResNet50 CV `94.35 ± 0.41%`, Test `95.00%`; ViT-S/16 CV `98.79 ± 0.27%`, Test `98.83%`).
  - **No LR scheduler** and **no gradient clipping**, matching Section 3.4 of the manuscript.
  - **On-the-fly augmentation active** (`RandomHorizontalFlip(0.5)`, `RandomRotation(20)`, `RandomAffine(scale=(0.9,1.1), shear=10)`, `ColorJitter(0.2, 0.2)`).
  - **ViT-S/16 loaded via `timm`** (`vit_small_patch16_224.augreg_in21k_ft_in1k`, no dropout, blocks 0–1 frozen -> `21.67M` total / `18.12M` trainable parameters), with weight equivalence verified against `WinKawaks/vit-small-patch16-224` (`src/verify_vit_checkpoint.py`, `logs_and_metrics/vit_checkpoint_check.json`).
- **Rotation-Invariant Near-Duplicate Audit (`src/exp4_duplicate_audit.py`, `src/train_clean_grouped.py`)**:
  - Dihedral-averaged ResNet50 embeddings + class-wise k-NN cosine + ORB/RANSAC geometric verification + union-find clustering (`1,779` groups) and `StratifiedGroupKFold` benchmark (`data_splits/exp4_grouped_split.csv`).
- **Patient-Grouped LungHist700 Protocol Without Test Leakage (`src/exp7_lunghist_finetune.py`, `src/exp1_zeroshot_ensemble.py`)**:
  - Replaced the `v1.0.0` `train_eval_grouped_kfold.py` script cleanly: uses `data_splits/folds_patient.csv` (136–142 images/fold across 45 patients), **inner patient-grouped validation (~15% patients)**, `patience = 8` on inner validation loss, and **no LR scheduler**, producing exact Table 6 OOF accuracies (`68.89% / 70.77% / 69.46%`), plus per-magnification (20× / 40×), image-level leakage ablation, and Durkalski clustered McNemar tests (`src/lunghist_stats.py`).
- **XAI Faithfulness Evaluation (`src/exp2_faithfulness.py`)**:
  - Added per-model independent random baselines, `K=10` & `K=20`, three replacement values (`grey`, `blur`, `tissue_mean`), raw logit sensitivity, and paired Wilcoxon signed-rank tests (`logs_and_metrics/exp2_faithfulness/`).
- **Computational Profiling & Complete Raw Telemetry (`src/exp3_compute_benchmark.py`, `logs_and_metrics/`)**:
  - Separated synthetic GPU-only throughput/latency/VRAM from wall-clock training logs; archived complete `run_all.log`, `environment.txt` (with SHA-256 checksums), `tables_for_manuscript.md`, and regenerated 300 DPI manuscript figures (`scripts/generate_manuscript_figures.py`).

## [1.0.0] - 2026-10-09
- Initial pre-revision replication package (archived at git tag `v1.0.0`).
