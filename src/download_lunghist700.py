"""Downloads and extracts LungHist700 dataset from the official Figshare repository.

Figshare DOI: 10.6084/m9.figshare.25459174
Article ID: 25459174 (File ID: 45206104, Size: ~639.3 MB)
Eliminates the need for manual file uploads to RunPod.
"""
import os
import shutil
import subprocess
import sys
import urllib.request
import zipfile

from common import get_lunghist_root

FIGSHARE_URL = "https://ndownloader.figshare.com/files/45206104"


def download_with_progress(url, dest_path):
    print(f"Downloading LungHist700 from Figshare: {url}")
    print(f"Destination: {dest_path}")
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
    req = urllib.request.Request(url, headers=headers)

    with urllib.request.urlopen(req) as resp:
        total_size = int(resp.headers.get("Content-Length", 0))
        chunk_size = 1024 * 1024  # 1MB chunks
        downloaded = 0

        with open(dest_path, "wb") as f:
            while True:
                chunk = resp.read(chunk_size)
                if not chunk:
                    break
                f.write(chunk)
                downloaded += len(chunk)
                if total_size > 0:
                    pct = (downloaded / total_size) * 100.0
                    mb_down = downloaded / (1024 * 1024)
                    mb_tot = total_size / (1024 * 1024)
                    sys.stdout.write(f"\r  Progress: {mb_down:.1f}/{mb_tot:.1f} MB ({pct:.1f}%)")
                    sys.stdout.flush()

    print("\n[OK] Download completed successfully!")


def extract_archive(archive_path, target_dir):
    os.makedirs(target_dir, exist_ok=True)
    print(f"Extracting {archive_path} to {target_dir}...")

    # 1. If it's a zip file
    if archive_path.endswith(".zip"):
        try:
            with zipfile.ZipFile(archive_path, "r") as zf:
                zf.extractall(target_dir)
            print("[OK] Zip extraction successful!")
            return True
        except Exception as e:
            print(f"[Warn] Zip extraction error: {e}")

    # 2. Try tar (modern libarchive handles RAR)
    try:
        res = subprocess.run(["tar", "-xf", archive_path, "-C", target_dir], capture_output=True)
        if res.returncode == 0:
            print("[OK] Archive extracted with tar!")
            return True
    except Exception:
        pass

    # 3. Try 7z
    try:
        res = subprocess.run(["7z", "x", archive_path, f"-o{target_dir}", "-y"], capture_output=True)
        if res.returncode == 0:
            print("[OK] Archive extracted with 7z!")
            return True
    except Exception:
        pass

    # 4. Try unar
    try:
        res = subprocess.run(["unar", archive_path, "-o", target_dir, "-f"], capture_output=True)
        if res.returncode == 0:
            print("[OK] Archive extracted with unar!")
            return True
    except Exception:
        pass

    # 5. Try unrar
    try:
        res = subprocess.run(["unrar", "x", "-o+", archive_path, f"{target_dir}/"], capture_output=True)
        if res.returncode == 0:
            print("[OK] Archive extracted with unrar!")
            return True
    except Exception:
        pass

    return False


def main():
    # 1. Check if already available
    existing_root = get_lunghist_root()
    if existing_root and os.path.isdir(os.path.join(existing_root, "aca_bd")):
        print(f"[Found] LungHist700 is already extracted and available at: {existing_root}")
        with open("lunghist_root.txt", "w") as f:
            f.write(existing_root)
        return

    print("LungHist700 dataset not found in active directory. Preparing dataset...")

    # 2. Search for existing archives in common locations
    candidate_archives = [
        "data/LungHist700.rar",
        "LungHist700.rar",
        "data/LungHist700.zip",
        "LungHist700.zip",
        "../LungHist700.zip",
        "../LungHist700.rar",
        "../../LungHist700.zip",
        "../../LungHist700.rar",
        "../../tahap_7/external_validation/data/LungHist700.rar",
        "/workspace/LungHist700.zip",
        "/workspace/LungHist700.rar",
        "/workspace/data/LungHist700.zip",
        "/workspace/data/LungHist700.rar"
    ]

    found_archive = None
    for cand in candidate_archives:
        if os.path.isfile(cand) and os.path.getsize(cand) > 50 * 1024 * 1024:
            found_archive = cand
            print(f"[Found Local Archive] {cand} ({os.path.getsize(cand)/(1024*1024):.1f} MB)")
            break

    # 3. If no local archive, download from official Figshare repository
    if not found_archive:
        os.makedirs("data", exist_ok=True)
        rar_dest = "data/LungHist700.rar"
        try:
            download_with_progress(FIGSHARE_URL, rar_dest)
            found_archive = rar_dest
        except Exception as e:
            print(f"[Error] Failed to download LungHist700 from Figshare: {e}")
            print("Fallback to curl...")
            curl_cmd = [
                "curl", "-L", "-A", "Mozilla/5.0",
                "--retry", "5",
                FIGSHARE_URL, "-o", rar_dest
            ]
            subprocess.run(curl_cmd, check=True)
            found_archive = rar_dest

    # 4. Extract archive
    target_extract = "data/LungHist700"
    os.makedirs(target_extract, exist_ok=True)
    extracted = extract_archive(found_archive, target_extract)

    # 5. Locate resolved root
    resolved_root = get_lunghist_root()
    if resolved_root and os.path.isdir(os.path.join(resolved_root, "aca_bd")):
        print(f"\n>>> [SUCCESS] LungHist700 ready at: {resolved_root}")
        with open("lunghist_root.txt", "w") as f:
            f.write(resolved_root)
    else:
        found = False
        for search_base in ["data", "/workspace"]:
            if not os.path.isdir(search_base):
                continue
            for root, dirs, _ in os.walk(search_base):
                if "aca_bd" in dirs:
                    final_p = root if root.endswith("/") else root + "/"
                    print(f">>> [SUCCESS] Discovered LungHist700 at: {final_p}")
                    with open("lunghist_root.txt", "w") as f:
                        f.write(final_p)
                    found = True
                    break
            if found:
                break
        if not found:
            raise RuntimeError(f"LungHist700 extraction failed (extracted={extracted}); directory 'aca_bd' not found!")


if __name__ == "__main__":
    main()
