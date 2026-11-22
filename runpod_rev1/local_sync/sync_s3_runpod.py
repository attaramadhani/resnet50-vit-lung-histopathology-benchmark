"""RunPod S3-Compatible API Sync Tool (Cara 1 - Zero GPU Idle Cost during Upload/Download).

Uses RunPod Network Volume S3 endpoint (e.g. https://s3api-eu-ro-1.runpod.io/) so you can:
  1. Upload C:\\RunPod_CMPBU_Rev1\\old_ckpts_for_runpod.zip (1.69 GB) BEFORE deploying the GPU Pod ($0 GPU cost).
  2. After training completes and you TERMINATE the RTX 4090 Pod ($0 GPU cost), download ALL results
     (metrics, logs, preds, and every .pt/.pth checkpoint part) into C:\\RunPod_CMPBU_Rev1\\,
     extract them, copy lightweight *_metrics_logs_preds.zip to repo results_rev1\\_archives\\,
     and verify every SHA-256 against MANIFEST_SHA256.txt before deleting the Network Volume!

Requires: pip install boto3
Set env vars (or pass flags):
  RUNPOD_S3_ACCESS_KEY  (from RunPod Console -> Settings -> S3 API Keys)
  RUNPOD_S3_SECRET_KEY
"""
import argparse
import glob
import os
import shutil
import subprocess
import sys
import zipfile


def make_client(region, access_key, secret_key):
    try:
        import boto3
    except ImportError:
        print("boto3 belum terpasang. Jalankan: pip install boto3")
        sys.exit(2)
    endpoint = f"https://s3api-{region}.runpod.io/"
    return boto3.client(
        "s3",
        region_name=region,
        endpoint_url=endpoint,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
    )


def upload_old_ckpts(client, volume_id, local_zip, remote_key):
    if not os.path.isfile(local_zip):
        raise FileNotFoundError(f"Tidak ditemukan: {local_zip}. Jalankan prepare_old_ckpts_for_runpod.py dulu.")
    sz_gb = os.path.getsize(local_zip) / (1024 ** 3)
    print(f"[S3 Upload] {local_zip} ({sz_gb:.2f} GB) -> s3://{volume_id}/{remote_key}")
    client.upload_file(local_zip, volume_id, remote_key)
    print("[OK] Upload selesai! File sudah tersedia di /workspace/old_ckpts_for_runpod.zip saat Pod dibuat.")


def list_all_objects(client, volume_id, prefix):
    items = []
    paginator = client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=volume_id, Prefix=prefix):
        for obj in page.get("Contents", []):
            items.append((obj["Key"], int(obj["Size"])))
    return items


def download_results(client, volume_id, prefix, local_backup_dir, repo_archives_dir):
    archives_dir = os.path.join(local_backup_dir, "_archives")
    extracted_dir = os.path.join(local_backup_dir, "results_rev1")
    os.makedirs(archives_dir, exist_ok=True)
    os.makedirs(extracted_dir, exist_ok=True)
    os.makedirs(repo_archives_dir, exist_ok=True)

    print(f"[S3 List] Memindai s3://{volume_id}/ (prefix={prefix}) ...")
    objs = []
    if prefix == "auto":
        all_objs = list_all_objects(client, volume_id, "")
        manifests = [k for k, _ in all_objs if k.endswith("_archives/MANIFEST_SHA256.txt")]
        if not manifests:
            print("[ERROR] MANIFEST_SHA256.txt tidak ditemukan di seluruh bucket S3! Pastikan package_rev1.py sudah selesai di RunPod.")
            sys.exit(1)
        # Pick the matching _archives folder automatically regardless of repo folder name on RunPod
        detected_prefix = os.path.dirname(manifests[0]).replace("\\", "/") + "/"
        print(f"[Auto-Detect] Folder arsip S3 ditemukan: {detected_prefix}")
        objs = [(k, s) for k, s in all_objs if k.startswith(detected_prefix)]
    else:
        objs = list_all_objects(client, volume_id, prefix)
    if not objs:
        print(f"[ERROR] Tidak ada file ditemukan di s3://{volume_id}/{prefix}")
        sys.exit(1)

    for key, r_size in objs:
        fn = os.path.basename(key)
        if not fn:
            continue
        dst = os.path.join(archives_dir, fn)
        if os.path.isfile(dst) and os.path.getsize(dst) == r_size:
            print(f"  [Skip - Exact Size Match] {fn} ({r_size} bytes)")
        else:
            gb_txt = f"{r_size / (1024**3):.2f} GB" if r_size >= 1024 ** 3 else f"{r_size / (1024**2):.2f} MB"
            print(f"  [Download] {fn} ({gb_txt})...")
            client.download_file(volume_id, key, dst)

        if fn.endswith("_metrics_logs_preds.zip") or fn in ("MANIFEST_SHA256.txt", "COMPLETENESS_AUDIT.txt"):
            shutil.copy2(dst, os.path.join(repo_archives_dir, fn))

    print(f"\n[Extract] Mengekstrak seluruh arsip ke {extracted_dir} ...")
    for zp in sorted(glob.glob(os.path.join(archives_dir, "*.zip"))):
        print("  Extracting:", os.path.basename(zp))
        with zipfile.ZipFile(zp, "r") as zf:
            zf.extractall(extracted_dir)

    verify_py = os.path.join(os.path.dirname(os.path.abspath(__file__)), "verify_local_backup.py")
    manifest = os.path.join(archives_dir, "MANIFEST_SHA256.txt")
    print("\n[Verify] Menjalankan verifikasi SHA-256 terhadap MANIFEST_SHA256.txt...")
    rc = subprocess.call([sys.executable, verify_py, "--manifest", manifest, "--results-dir", extracted_dir])
    if rc == 0:
        print("\n[COMPLETE] Semua hasil & .pt/.pth aman di lokal! Network Volume di RunPod sekarang aman untuk dihapus.")
    sys.exit(rc)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["upload_old_ckpts", "download_results"])
    ap.add_argument("--volume-id", required=True, help="RunPod Network Volume ID (acts as S3 Bucket)")
    ap.add_argument("--region", default="eu-ro-1", help="RunPod Datacenter ID (default: eu-ro-1)")
    ap.add_argument("--access-key", default=os.environ.get("RUNPOD_S3_ACCESS_KEY", ""))
    ap.add_argument("--secret-key", default=os.environ.get("RUNPOD_S3_SECRET_KEY", ""))
    ap.add_argument("--local-old-zip", default=r"C:\RunPod_CMPBU_Rev1\old_ckpts_for_runpod.zip")
    ap.add_argument("--local-backup-dir", default=r"C:\RunPod_CMPBU_Rev1")
    ap.add_argument("--remote-prefix", default="auto", help="S3 key prefix or 'auto' to auto-detect repo folder name")
    args = ap.parse_args()

    if not args.access_key or not args.secret_key:
        raise ValueError("Set RUNPOD_S3_ACCESS_KEY & RUNPOD_S3_SECRET_KEY atau pass --access-key dan --secret-key.")

    client = make_client(args.region, args.access_key, args.secret_key)
    if args.mode == "upload_old_ckpts":
        upload_old_ckpts(client, args.volume_id, args.local_old_zip, "old_ckpts_for_runpod.zip")
    else:
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        repo_archives = os.path.join(repo_root, "results_rev1", "_archives")
        download_results(client, args.volume_id, args.remote_prefix, args.local_backup_dir, repo_archives)


if __name__ == "__main__":
    main()
