"""Extract ONLY the 20 v2 checkpoints needed on RunPod (~1.8 GB vs 11.34 GB full archive).

Needed checkpoints:
  - runs/resnet50_main/fold{0..4}.pt    (WP06a zero-shot & WP06c/e refs)
  - runs/vit_s_main/fold{0..4}.pt       (WP06a zero-shot & WP06c ViT LC-init)
  - runs/clean_resnet50/fold{0..4}.pt   (WP05 old G1 per-image predictions)
  - runs/clean_vit_s16/fold{0..4}.pt    (WP05 old G2 per-image predictions)
"""
import argparse
import os
import zipfile

TARGET_PREFIXES = (
    "runs/resnet50_main/",
    "runs/vit_s_main/",
    "runs/clean_resnet50/",
    "runs/clean_vit_s16/",
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--src-zip",
        default=r"..\Hasil Eksperimen new\checkpoints.zip",
        help="Path to the 11.3 GB checkpoints.zip archive",
    )
    ap.add_argument(
        "--out-zip",
        default=r"C:\RunPod_CMPBU_Rev1\old_ckpts_for_runpod.zip",
        help="Output compact archive (~1.8 GB) stored outside OneDrive for fast SCP upload to RunPod",
    )
    args = ap.parse_args()

    if not os.path.exists(args.src_zip):
        raise FileNotFoundError(f"Source checkpoints.zip not found: {args.src_zip}")

    os.makedirs(os.path.dirname(args.out_zip) or ".", exist_ok=True)
    count = 0
    total_bytes = 0
    with zipfile.ZipFile(args.src_zip, "r") as zin, zipfile.ZipFile(args.out_zip, "w", zipfile.ZIP_STORED) as zout:
        for info in zin.infolist():
            if info.is_dir():
                continue
            if info.filename.startswith(TARGET_PREFIXES) and info.filename.lower().endswith((".pt", ".pth")):
                zout.writestr(info, zin.read(info.filename))
                count += 1
                total_bytes += info.file_size
                print(f"  + {info.filename} ({info.file_size / (1024*1024):.1f} MB)")

    print(f"\n[OK] Packed {count} checkpoints ({total_bytes / (1024**3):.2f} GB) -> {args.out_zip}")
    if count != 20:
        print(f"[WARNING] Expected 20 checkpoints, found {count}. Check source archive layout!")


if __name__ == "__main__":
    main()
