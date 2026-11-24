"""verify_local_backup.py - Strict SHA-256 & byte-size check before stopping/terminating RunPod.

Verifies every single file listed in MANIFEST_SHA256.txt (including every .pt/.pth checkpoint,
predictions CSV, metrics JSON, history CSV, config YAML, and log) exists intact on local disk.
Exits with code 0 ONLY if 0 files are missing and 0 files are corrupted.
"""
import argparse
import hashlib
import os
import sys


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True, help="Path to MANIFEST_SHA256.txt")
    ap.add_argument("--results-dir", required=True, help="Local extracted results_rev1 directory")
    args = ap.parse_args()

    if not os.path.exists(args.manifest):
        print(f"[ERROR] Manifest not found: {args.manifest}")
        sys.exit(2)

    missing = []
    size_mismatch = []
    hash_mismatch = []
    verified_total = 0
    verified_ckpts = 0
    total_bytes = 0

    with open(args.manifest, "r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f if ln.strip()]

    for ln in lines:
        parts = ln.split(None, 2)
        if len(parts) != 3:
            continue
        exp_hash, exp_size_str, rel_path = parts
        exp_size = int(exp_size_str)
        local_p = os.path.join(args.results_dir, rel_path.replace("/", os.sep))

        if not os.path.isfile(local_p):
            missing.append(rel_path)
            continue
        actual_size = os.path.getsize(local_p)
        if actual_size != exp_size:
            size_mismatch.append((rel_path, exp_size, actual_size))
            continue
        actual_hash = sha256_file(local_p)
        if actual_hash != exp_hash:
            hash_mismatch.append(rel_path)
            continue

        verified_total += 1
        total_bytes += actual_size
        if rel_path.lower().endswith((".pt", ".pth")):
            verified_ckpts += 1

    print("=" * 72)
    print(f"  LOCAL BACKUP VERIFICATION SUMMARY")
    print(f"  Manifest entries     : {len(lines)}")
    print(f"  Verified intact      : {verified_total} ({total_bytes / (1024**3):.2f} GB)")
    print(f"  Verified .pt/.pth    : {verified_ckpts}")
    print(f"  Missing files        : {len(missing)}")
    print(f"  Size mismatches      : {len(size_mismatch)}")
    print(f"  SHA-256 mismatches   : {len(hash_mismatch)}")
    print("=" * 72)

    if missing:
        print("\n[MISSING FILES] (first 20):")
        for m in missing[:20]:
            print("  -", m)
    if size_mismatch:
        print("\n[SIZE MISMATCHES] (first 20):")
        for r, es, ac in size_mismatch[:20]:
            print(f"  - {r}: expected {es}, got {ac}")
    if hash_mismatch:
        print("\n[SHA-256 MISMATCHES] (first 20):")
        for h_item in hash_mismatch[:20]:
            print("  -", h_item)

    if not missing and not size_mismatch and not hash_mismatch:
        print("\n[SAFE TO TERMINATE RUNPOD POD] Seluruh hasil eksperimen & file .pt/.pth terverifikasi lengkap di lokal!")
        sys.exit(0)
    else:
        print("\n[DO NOT TERMINATE RUNPOD POD YET] Masih ada file yang belum tersinkron atau rusak!")
        sys.exit(1)


if __name__ == "__main__":
    main()
