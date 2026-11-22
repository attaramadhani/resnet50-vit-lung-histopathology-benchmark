<#
.SYNOPSIS
  Zero-loss RunPod -> Local Windows backup & verification script (CMPBUP-D-26-00450).
  Saves heavy .pt/.pth weights OUTSIDE OneDrive (default C:\RunPod_CMPBU_Rev1) and copies
  lightweight *_metrics_logs_preds.zip into the local repo results_rev1\_archives\.
#>
param(
    [Parameter(Mandatory=$true)][string]$HostIp,
    [Parameter(Mandatory=$true)][int]$Port,
    [string]$User = "root",
    [string]$KeyPath = "$env:USERPROFILE\.ssh\id_ed25519",
    [string]$RemoteRepo = "/workspace/resnet50-vit-lung-histopathology-benchmark",
    [string]$LocalBackupDir = "C:\RunPod_CMPBU_Rev1",
    [switch]$IncludeCheckpoints = $true
)

$ErrorActionPreference = "Stop"

$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$LocalRepoArchives = Join-Path $RepoRoot "results_rev1\_archives"
$BackupArchives = Join-Path $LocalBackupDir "_archives"
$BackupExtracted = Join-Path $LocalBackupDir "results_rev1"

New-Item -ItemType Directory -Force -Path $LocalRepoArchives | Out-Null
New-Item -ItemType Directory -Force -Path $BackupArchives | Out-Null
New-Item -ItemType Directory -Force -Path $BackupExtracted | Out-Null

$SshOpts = @("-i", $KeyPath, "-p", "$Port", "-o", "StrictHostKeyChecking=no", "-o", "ServerAliveInterval=30")
$ScpOpts = @("-i", $KeyPath, "-P", "$Port", "-o", "StrictHostKeyChecking=no", "-o", "ServerAliveInterval=30")

Write-Host "========================================================================"
Write-Host "  1. Refreshing package_rev1.py manifest & archives on RunPod..."
Write-Host "========================================================================"
& ssh @SshOpts "$User@$HostIp" "cd $RemoteRepo && python runpod_rev1/package_rev1.py --all"

Write-Host "`n========================================================================"
Write-Host "  2. Downloading MANIFEST_SHA256.txt, audit, and lightweight WP zips..."
Write-Host "========================================================================"
& scp @ScpOpts "$User@${HostIp}:${RemoteRepo}/results_rev1/_archives/MANIFEST_SHA256.txt" "$BackupArchives\"
& scp @ScpOpts "$User@${HostIp}:${RemoteRepo}/results_rev1/_archives/COMPLETENESS_AUDIT.txt" "$BackupArchives\"
Copy-Item -Force "$BackupArchives\MANIFEST_SHA256.txt" "$LocalRepoArchives\"
Copy-Item -Force "$BackupArchives\COMPLETENESS_AUDIT.txt" "$LocalRepoArchives\"

$LightFiles = & ssh @SshOpts "$User@$HostIp" "ls -1 ${RemoteRepo}/results_rev1/_archives/*_metrics_logs_preds.zip 2>/dev/null || true"
foreach ($rf in $LightFiles) {
    $rf = $rf.Trim()
    if (-not $rf) { continue }
    $fname = [System.IO.Path]::GetFileName($rf)
    Write-Host "  Downloading $fname..."
    & scp @ScpOpts "$User@${HostIp}:$rf" "$BackupArchives\"
    Copy-Item -Force "$BackupArchives\$fname" "$LocalRepoArchives\"
}

if ($IncludeCheckpoints) {
    Write-Host "`n========================================================================"
    Write-Host "  3. Downloading ALL checkpoint parts (.pt/.pth) to $BackupArchives..."
    Write-Host "     (Stored on $LocalBackupDir OUTSIDE OneDrive to avoid cloud sync bloat)"
    Write-Host "========================================================================"
    $RemoteList = & ssh @SshOpts "$User@$HostIp" "stat -c '%s %n' ${RemoteRepo}/results_rev1/_archives/*_checkpoints_part*.zip 2>/dev/null || true"
    foreach ($line in $RemoteList) {
        $line = $line.Trim()
        if (-not $line) { continue }
        $parts = $line -split "\s+", 2
        $rSize = [int64]$parts[0]
        $rPath = $parts[1]
        $fname = [System.IO.Path]::GetFileName($rPath)
        $lPath = Join-Path $BackupArchives $fname
        if ((Test-Path $lPath) -and ((Get-Item $lPath).Length -eq $rSize)) {
            Write-Host "  [Skip - Exact Size Match] $fname ($rSize bytes)"
        } else {
            Write-Host "  [Download] $fname ($([math]::Round($rSize / 1GB, 2)) GB)..."
            & scp @ScpOpts "$User@${HostIp}:$rPath" "$BackupArchives\"
        }
    }
}

Write-Host "`n========================================================================"
Write-Host "  4. Extracting archives into $BackupExtracted..."
Write-Host "========================================================================"
python -c "
import glob, os, zipfile
src = r'$BackupArchives'
dst = r'$BackupExtracted'
for zp in sorted(glob.glob(os.path.join(src, '*.zip'))):
    print('Extracting:', os.path.basename(zp))
    with zipfile.ZipFile(zp, 'r') as zf:
        zf.extractall(dst)
"

Write-Host "`n========================================================================"
Write-Host "  5. Verifying EVERY file & .pt/.pth checksum against MANIFEST_SHA256.txt..."
Write-Host "========================================================================"
$VerifyScript = Join-Path $PSScriptRoot "verify_local_backup.py"
python $VerifyScript --manifest "$BackupArchives\MANIFEST_SHA256.txt" --results-dir "$BackupExtracted"
