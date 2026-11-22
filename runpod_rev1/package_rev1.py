"""package_rev1.py - Package Revision R1 results & checkpoints with zero-loss SHA-256 manifest.

Creates inside results_rev1/_archives/:
  1. <WP>_metrics_logs_preds.zip : lightweight deliverable for Dr. Wahyudi (CSVs, JSONs, YAMLs, PNGs, logs, preds)
  2. <WP>_checkpoints_partNN.zip : all *.pt and *.pth files split into <= 3.8 GB ZIP_STORED parts
  3. MANIFEST_SHA256.txt         : byte size + SHA-256 checksum of EVERY file in results_rev1/
  4. COMPLETENESS_AUDIT.txt      : expected vs actual counts per WP (never miss a file or .pt/.pth!)
"""
import argparse
import hashlib
import json
import os
import zipfile

MAX_PART_BYTES = int(3.8 * 1024 * 1024 * 1024)
RESULTS_ROOT = "results_rev1"
ARCHIVES_DIR = os.path.join(RESULTS_ROOT, "_archives")

def get_expected_counts(paket="C"):
    paket = paket.upper()
    counts = {
        "WP02":  {"files": ["wp02_grid.csv", "wp02_grid.md", "wp02_pairs.csv", "wp02_pool.csv"], "min_ckpts": 0, "min_preds": 0},
        "WP03":  {"files": ["exp3_vit_s16_unfrozen.json"], "min_ckpts": 60, "min_preds": 12},
        "WP05":  {"files": ["wp05_g1_g2_summary.json", "G1_clean_resnet50_ensemble_test_preds.csv", "G2_clean_vit_s16_ensemble_test_preds.csv"], "min_ckpts": 0, "min_preds": 2},
        "WP04c": {"files": [], "min_ckpts": 60, "min_preds": 12},
        "WP06":  {"files": ["wp06a/wp06a_summary.json", "wp06d/wp06d_summary.json", "wp06e/wp06e_orig_protocol_metrics.json", "wp06f/wp06f_domain_shift_metrics.json"], "min_ckpts": 20, "min_preds": 0},
    }
    if paket == "B":
        counts["WP04a"] = {"files": [], "min_ckpts": 80, "min_preds": 16}
    elif paket == "C":
        counts["WP04a"] = {"files": [], "min_ckpts": 160, "min_preds": 32}
        counts["WP04b"] = {"files": [], "min_ckpts": 140, "min_preds": 28}
    return counts


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def collect_wp_files(wp_name):
    wp_dir = os.path.join(RESULTS_ROOT, wp_name)
    light_files = []
    ckpt_files = []
    other_heavy = []

    if os.path.isdir(wp_dir):
        for r, _, fs in os.walk(wp_dir):
            for fn in sorted(fs):
                full = os.path.join(r, fn)
                rel = os.path.relpath(full, RESULTS_ROOT).replace("\\", "/")
                low = fn.lower()
                if low.endswith(".pt") or low.endswith(".pth"):
                    ckpt_files.append((full, rel))
                elif low.endswith(".npz"):
                    other_heavy.append((full, rel))
                else:
                    light_files.append((full, rel))

    # Include corresponding log + top-level provenance files
    log_dir = os.path.join(RESULTS_ROOT, "logs")
    if os.path.isdir(log_dir):
        tag = wp_name.lower()
        for fn in sorted(os.listdir(log_dir)):
            if tag in fn.lower():
                full = os.path.join(log_dir, fn)
                rel = os.path.relpath(full, RESULTS_ROOT).replace("\\", "/")
                light_files.append((full, rel))
    for top in ["environment_rev1.txt", "scripts_sha256_rev1.txt"]:
        full = os.path.join(RESULTS_ROOT, top)
        if os.path.isfile(full):
            light_files.append((full, top))

    return light_files, ckpt_files, other_heavy


def package_wp(wp_name):
    os.makedirs(ARCHIVES_DIR, exist_ok=True)
    light_files, ckpt_files, other_heavy = collect_wp_files(wp_name)
    if not light_files and not ckpt_files:
        print(f"[Skip] {wp_name}: belum ada file hasil di {RESULTS_ROOT}/{wp_name}.")
        return

    # 1. Lightweight zip for Pak Wahyudi (+ .npz features inside light? No, features are large, keep optional separate or inside checkpoints)
    light_zip = os.path.join(ARCHIVES_DIR, f"{wp_name}_metrics_logs_preds.zip")
    with zipfile.ZipFile(light_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        for full, rel in light_files:
            zf.write(full, rel)
    l_mb = os.path.getsize(light_zip) / (1024 * 1024)
    print(f"[OK] {light_zip}: {len(light_files)} files ({l_mb:.2f} MB)")

    # 2. Chunked checkpoint (+ optional .npz) zip(s) so EVERY .pt/.pth is safely split <= 3.8 GB/part
    heavy_all = ckpt_files + other_heavy
    if heavy_all:
        # Remove old parts for this WP before rebuilding cleanly
        for fn in os.listdir(ARCHIVES_DIR):
            if fn.startswith(f"{wp_name}_checkpoints_part") and fn.endswith(".zip"):
                os.remove(os.path.join(ARCHIVES_DIR, fn))

        part = 1
        cur_bytes = 0
        cur_list = []

        def flush_part(items, p_idx):
            p_path = os.path.join(ARCHIVES_DIR, f"{wp_name}_checkpoints_part{p_idx:02d}.zip")
            with zipfile.ZipFile(p_path, "w", zipfile.ZIP_STORED) as zf:
                for full, rel in items:
                    zf.write(full, rel)
            gb = os.path.getsize(p_path) / (1024 ** 3)
            print(f"[OK] {p_path}: {len(items)} weight/feature files ({gb:.2f} GB)")

        for full, rel in heavy_all:
            sz = os.path.getsize(full)
            if cur_list and (cur_bytes + sz > MAX_PART_BYTES):
                flush_part(cur_list, part)
                part += 1
                cur_bytes = 0
                cur_list = []
            cur_list.append((full, rel))
            cur_bytes += sz
        if cur_list:
            flush_part(cur_list, part)


def write_manifest_and_audit(paket="C"):
    os.makedirs(ARCHIVES_DIR, exist_ok=True)
    manifest_path = os.path.join(ARCHIVES_DIR, "MANIFEST_SHA256.txt")
    rows = []
    for r, _, fs in os.walk(RESULTS_ROOT):
        norm_r = os.path.normpath(r)
        if os.path.normpath(ARCHIVES_DIR) in norm_r:
            continue
        for fn in sorted(fs):
            full = os.path.join(r, fn)
            rel = os.path.relpath(full, RESULTS_ROOT).replace("\\", "/")
            sz = os.path.getsize(full)
            digest = sha256_file(full)
            rows.append((digest, sz, rel))

    with open(manifest_path, "w", encoding="utf-8") as f:
        for digest, sz, rel in rows:
            f.write(f"{digest}  {sz:14d}  {rel}\n")

    total_gb = sum(sz for _, sz, _ in rows) / (1024 ** 3)
    n_ckpts = sum(1 for _, _, rel in rows if rel.lower().endswith((".pt", ".pth")))
    print(f"\n[OK] {manifest_path}: {len(rows)} files indexed ({n_ckpts} .pt/.pth checkpoints, total {total_gb:.2f} GB)")

    expected_counts = get_expected_counts(paket)
    audit_lines = [
        f"=== REVISION R1 COMPLETENESS AUDIT (PAKET {paket.upper()}) ===",
        f"Total indexed files : {len(rows)}",
        f"Total .pt/.pth files: {n_ckpts}",
        f"Total size (raw)    : {total_gb:.2f} GB",
        "",
    ]
    for wp, spec in expected_counts.items():
        wp_dir = os.path.join(RESULTS_ROOT, wp)
        if not os.path.isdir(wp_dir):
            audit_lines.append(f"[{wp}] NOT STARTED")
            continue
        wp_ckpts = [rel for _, _, rel in rows if rel.startswith(wp + "/") and rel.lower().endswith((".pt", ".pth"))]
        wp_preds = [rel for _, _, rel in rows if rel.startswith(wp + "/") and ("test_preds.csv" in rel or "_oof.csv" in rel)]
        missing_req = [req for req in spec["files"] if not os.path.exists(os.path.join(wp_dir, req))]
        status = "OK" if (len(wp_ckpts) >= spec["min_ckpts"] and len(wp_preds) >= spec.get("min_preds", 0) and not missing_req) else "INCOMPLETE"
        audit_lines.append(
            f"[{wp}] {status:10s} | .pt/.pth={len(wp_ckpts)} (min {spec['min_ckpts']}) | preds/oof={len(wp_preds)} | missing={missing_req}"
        )

    audit_path = os.path.join(ARCHIVES_DIR, "COMPLETENESS_AUDIT.txt")
    with open(audit_path, "w", encoding="utf-8") as f:
        f.write("\n".join(audit_lines) + "\n")
    print("\n" + "\n".join(audit_lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wp", default=None, help="Specific WP name (e.g. WP02, WP03, WP04a, WP04b, WP04c, WP05, WP06)")
    ap.add_argument("--paket", default="C", choices=["A", "B", "C", "a", "b", "c"], help="Expected Paket completeness profile")
    ap.add_argument("--all", action="store_true", help="Package all existing WP folders")
    args = ap.parse_args()

    if args.wp:
        package_wp(args.wp)
    elif args.all or True:
        for wp in ["WP01", "WP02", "WP03", "WP04a", "WP04b", "WP04c", "WP05", "WP06"]:
            if os.path.isdir(os.path.join(RESULTS_ROOT, wp)):
                package_wp(wp)

    write_manifest_and_audit(args.paket)


if __name__ == "__main__":
    main()
