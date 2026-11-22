# Changelog

## [2.0.0] - 2026-10-15 (Full Revision Package Aligned with CMPBUP-D-26-00450)

Replaces the initial `v1.0.0` draft artifacts with the unified revision pipeline executed on NVIDIA RTX 4090 (RunPod, Oct 14–15, 2026) and resolves all code–log–manuscript discrepancies:

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
