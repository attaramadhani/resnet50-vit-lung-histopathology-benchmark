"""Synchronizes all experimental output aliases across ./, runs/, and results/.

Guarantees 100% interoperability across different naming conventions used in various scripts:
  - Scenario directories: vit_s_main <-> vit_s16_main, resnet50_main <-> resnet50_adam, etc.
  - Experiment files: exp2_vit <-> exp2_vit_s16, e7_vit <-> e7_vit_s16, etc.
  - Mirrored storage: keeps copies in both root ./ and results/ so no script fails with FileNotFoundError.
"""
import glob
import os
import shutil

FOLDER_ALIASES = [
    ("runs/vit_s_main", ["runs/vit_s16_main", "runs/vit_s"]),
    ("runs/resnet50_main", ["runs/resnet50_adam", "runs/r50_main"]),
    ("runs/resnet50_sgdm", ["runs/resnet50_sgdm_1e4"]),
    ("runs/vit_s_sgdm", ["runs/vit_s16_sgdm", "runs/vit_s_sgdm_1e4"]),
    ("runs/resnet50_sgdm_lr_1e3", ["runs/resnet50_sgdm_1e3"]),
    ("runs/vit_s_sgdm_lr_1e3", ["runs/vit_s16_sgdm_lr_1e3", "runs/vit_s_sgdm_1e3"]),
    ("runs/clean_resnet50", ["runs/clean_r50"]),
    ("runs/clean_vit_s16", ["runs/clean_vit_s"])
]

FILE_PREFIX_ALIASES = [
    ("exp2_vit", "exp2_vit_s16"),
    ("exp3_vit_s16", "exp3_vit_s"),
    ("e7_vit", "e7_vit_s16"),
    ("e6_vit_image", "e6_vit_s16_image"),
    ("exp1_vit_main", "exp1_vit_s16_main"),
    ("exp1_vit_crop768", "exp1_vit_s16_crop768"),
    ("per_fold_val.csv", "per_fold.csv")
]


def sync_folders():
    for primary, aliases in FOLDER_ALIASES:
        # Determine which folder has content
        src_folder = None
        if os.path.isdir(primary) and len(os.listdir(primary)) > 0:
            src_folder = primary
        else:
            for alt in aliases:
                if os.path.isdir(alt) and len(os.listdir(alt)) > 0:
                    src_folder = alt
                    break

        if src_folder:
            targets = [primary] + aliases
            for target in targets:
                if target == src_folder:
                    continue
                os.makedirs(target, exist_ok=True)
                for item in os.listdir(src_folder):
                    s_path = os.path.join(src_folder, item)
                    t_path = os.path.join(target, item)
                    if os.path.isfile(s_path):
                        if not os.path.exists(t_path) or os.path.getmtime(s_path) > os.path.getmtime(t_path):
                            shutil.copy2(s_path, t_path)


def sync_file_prefixes():
    for p1, p2 in FILE_PREFIX_ALIASES:
        if p1.endswith(".csv"):
            # Direct file sync
            if os.path.isfile(p1) and not os.path.isfile(p2):
                shutil.copy2(p1, p2)
            elif os.path.isfile(p2) and not os.path.isfile(p1):
                shutil.copy2(p2, p1)
            continue

        # Pattern sync
        for f in glob.glob(f"{p1}*"):
            if os.path.isfile(f):
                equiv = f.replace(p1, p2, 1)
                if not os.path.exists(equiv) or os.path.getmtime(f) > os.path.getmtime(equiv):
                    shutil.copy2(f, equiv)

        for f in glob.glob(f"{p2}*"):
            if os.path.isfile(f):
                equiv = f.replace(p2, p1, 1)
                if not os.path.exists(equiv) or os.path.getmtime(f) > os.path.getmtime(equiv):
                    shutil.copy2(f, equiv)


def sync_to_results_directory():
    """Maintains a unified results/ directory mirroring all summary metrics and tables."""
    os.makedirs("results", exist_ok=True)
    patterns = [
        "*.csv",
        "*.json",
        "*.txt",
        "*.md"
    ]
    for pat in patterns:
        for f in glob.glob(pat):
            if os.path.isfile(f) and not f.startswith("results"):
                dest = os.path.join("results", os.path.basename(f))
                if not os.path.exists(dest) or os.path.getmtime(f) > os.path.getmtime(dest):
                    shutil.copy2(f, dest)


def main():
    sync_folders()
    sync_file_prefixes()
    sync_to_results_directory()
    print("[OK] Results directory and naming aliases synchronized successfully!")


if __name__ == "__main__":
    main()
