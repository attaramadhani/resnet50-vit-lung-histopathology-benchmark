# RunPod Revision R1 Setup, Paket A/B/C & S3 Zero-Loss Sync Guide (CMPBUP-D-26-00450)

## 1. Persiapan Lokal & Upload Awal via S3 API ($0 Biaya GPU)
1. **Pastikan arsip 20 checkpoint v2 sudah ada di lokal** (`C:\RunPod_CMPBU_Rev1\old_ckpts_for_runpod.zip`, 1,69 GB). Jika belum:
   ```powershell
   cd "C:\Users\Atta\OneDrive\Documents\Proyek Bapak Wahyudi\resnet50-vit-lung-histopathology-benchmark"
   python runpod_rev1\local_sync\prepare_old_ckpts_for_runpod.py
   ```
2. **Buat Network Volume di RunPod:**
   - Datacenter: **`EU-RO-1`** (mendukung **S3**)
   - Size: **`100 GB`** untuk **Paket C** (atau 60–80 GB untuk Paket A/B)
   - Catat **Network Volume ID** Anda.
3. **Buat S3 API Key** di RunPod Console (*Settings → S3 API Keys*), pasang paket `boto3` di lokal (`pip install boto3`), lalu upload `1,69 GB` ke Network Volume **sebelum** menyalakan GPU:
   ```powershell
   $env:RUNPOD_S3_ACCESS_KEY = "user_xxx"
   $env:RUNPOD_S3_SECRET_KEY = "rps_xxx"
   python runpod_rev1\local_sync\sync_s3_runpod.py --mode upload_old_ckpts --volume-id "<VOLUME_ID>" --region eu-ro-1
   ```

## 2. Deploy Pod RTX 4090 dari Network Volume (`EU-RO-1`)
1. Dari halaman Network Volume `EU-RO-1` Anda → klik **Deploy** → pilih **1× RTX 4090** + template **`PyTorch 2.8.0 · CUDA 12.8.1 · Ubuntu 24.04`**.
2. Pada *Edit Pod*, turunkan **Container Disk** menjadi **`20 GB`** (volume 100 GB otomatis ter-mount di `/workspace`).
3. SSH ke Pod dan clone repository (branch `rev1-runpod` atau `main`):
   ```bash
   cd /workspace
   git clone -b rev1-runpod https://github.com/attaramadhani/resnet50-vit-lung-histopathology-benchmark.git
   cd resnet50-vit-lung-histopathology-benchmark
   bash runpod_rev1/00_setup_runpod.sh
   ```

## 3. Opsi Paket & Sub-Eksperimen di `run_master_rev1.sh`
Selalu jalankan di dalam `tmux new -s cmpbu`:
```bash
# Paket C (Lengkap penuh WP02 - WP06, ~34-40 jam RTX 4090):
bash runpod_rev1/run_master_rev1.sh paket_c

# Paket B (Prioritas Pak Wahyudi: WP02, WP03, WP05, WP06, WP04a priority, WP04c):
bash runpod_rev1/run_master_rev1.sh paket_b

# Paket A (Inti metodologi tanpa WP04a/b):
bash runpod_rev1/run_master_rev1.sh paket_a

# Opsi granular satuan / lanjutan:
bash runpod_rev1/run_master_rev1.sh wp02
bash runpod_rev1/run_master_rev1.sh wp03
bash runpod_rev1/run_master_rev1.sh wp04a priority   # atau: all
bash runpod_rev1/run_master_rev1.sh wp04b
bash runpod_rev1/run_master_rev1.sh wp04c
bash runpod_rev1/run_master_rev1.sh wp06 all         # atau: a / b / c / d / e / f / a,c,f
```
> Jika Pod sempat terhenti tengah jalan, cukup jalankan ulang perintah `paket_c` (atau sub-WP terkait) — seluruh fold/eksperimen yang sudah selesai akan otomatis di-skip (`[Skip]`).

## 4. Terminate GPU Segera, Lalu Unduh ke Lokal via S3 API ($0 Biaya GPU)
1. Tunggu sampai `run_master_rev1.sh` menampilkan `[DONE] Master Rev1` dan cek isi `results_rev1/_archives/COMPLETENESS_AUDIT.txt` di RunPod berstatus **`OK`** semua.
2. **Langsung `Terminate` Pod RTX 4090** (tagihan `$0.89/jam` langsung berhenti, seluruh hasil tetap aman di Network Volume `EU-RO-1`).
3. Dari PowerShell lokal Anda, tarik seluruh hasil + `.pt`/`.pth` + verifikasi SHA-256 otomatis:
   ```powershell
   python runpod_rev1\local_sync\sync_s3_runpod.py --mode download_results --volume-id "<VOLUME_ID>" --region eu-ro-1
   ```
4. **Setelah muncul pesan:**  
   `[SAFE TO TERMINATE RUNPOD POD] Seluruh hasil eksperimen & file .pt/.pth terverifikasi lengkap di lokal!`  
   barulah **Delete Network Volume** di dashboard RunPod agar biaya storage berhenti total.
