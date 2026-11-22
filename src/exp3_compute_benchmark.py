"""Exp 3 - Computational profiling on NVIDIA RTX 4090 (Section 3.5, Table 4, Fig. 4).

Measures GPU-only computational metrics with synthetic tensors (no I/O overhead):
  * MACs (G) per 224x224 image via fvcore FlopCountAnalysis (or thop fallback)
  * Training throughput (img/s): forward + backward + optimizer step, FP16 autocast, batch 32, CUDA events
  * Latency (ms): per image for batch 32 and batch 1, after warm-up, measured via CUDA events
  * Peak GPU memory (GB) during training: torch.cuda.max_memory_allocated
  * Real training time: extracts time per epoch (train+val), epochs at stop, and total run duration from training logs

Usage:
  python exp3_compute_benchmark.py --arch resnet50
  python exp3_compute_benchmark.py --arch vit_s16
  python exp3_compute_benchmark.py --summarize-real-time
"""
import argparse
import glob
import json
import os
import subprocess
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from common import build_model, count_parameters


def get_gpu_info():
    if not torch.cuda.is_available():
        return "CPU"
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
            encoding="utf-8"
        )
        return out.strip()
    except Exception:
        return torch.cuda.get_device_name(0)


def summarize_real_time_logs():
    """Extracts real-world training times across the 70 benchmark runs."""
    history_files = glob.glob("runs/**/history.csv", recursive=True)
    if not history_files:
        print("[Info] No training logs (runs/**/history.csv) found yet.")
        return None

    records = []
    total_seconds = 0.0

    for hf in history_files:
        try:
            df = pd.read_csv(hf)
            sc_dir = os.path.dirname(hf)
            scenario = os.path.basename(os.path.dirname(sc_dir)) if "fold" in os.path.basename(sc_dir) else os.path.basename(sc_dir)
            fold = os.path.basename(sc_dir) if "fold" in os.path.basename(sc_dir) else "unknown"

            if "time_s" in df.columns:
                run_time = df["time_s"].sum()
                mean_ep_time = df["time_s"].mean()
            else:
                run_time = 0.0
                mean_ep_time = 0.0

            records.append({
                "scenario": scenario,
                "fold": fold,
                "epochs": len(df),
                "duration_seconds": run_time,
                "mean_epoch_seconds": mean_ep_time,
                "file": hf
            })
            total_seconds += run_time
        except Exception as e:
            continue

    df_real = pd.DataFrame(records)
    total_hrs = total_seconds / 3600.0

    summary = {
        "total_runs_analyzed": len(records),
        "total_duration_seconds": round(total_seconds, 2),
        "total_duration_hours": round(total_hrs, 2),
        "mean_epoch_seconds_overall": round(float(df_real["mean_epoch_seconds"].mean()), 2) if len(df_real) > 0 else 0.0,
        "mean_epochs_at_stop": round(float(df_real["epochs"].mean()), 2) if len(df_real) > 0 else 0.0,
        "by_scenario": {}
    }

    if "scenario" in df_real.columns and len(df_real) > 0:
        for sc, g in df_real.groupby("scenario"):
            summary["by_scenario"][sc] = {
                "runs": len(g),
                "mean_epochs": round(float(g["epochs"].mean()), 2),
                "mean_epoch_seconds": round(float(g["mean_epoch_seconds"].mean()), 2),
                "total_seconds": round(float(g["duration_seconds"].sum()), 2)
            }

    with open("real_training_time_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\n[OK] Real training time summarized: {len(records)} runs, total = {total_hrs:.2f} hours")
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", choices=["resnet50", "vit_s16", "vit_s"], default=None)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--freeze_vit", type=int, default=1, choices=[0, 1])
    ap.add_argument("--out", default=None)
    ap.add_argument("--summarize-real-time", action="store_true")
    args = ap.parse_args()

    if args.summarize_real_time or args.arch is None:
        summarize_real_time_logs()
        if args.arch is None:
            return

    arch_name = "vit_s16" if args.arch in ["vit_s16", "vit_s"] else "resnet50"
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    gpu_info = get_gpu_info()
    freeze_vit_flag = bool(args.freeze_vit)

    print(f"\n==========================================================================")
    print(f"  EXP 3: COMPUTATIONAL PROFILING (Table 4)")
    print(f"  Arch: {arch_name} | Freeze ViT B0-1: {freeze_vit_flag} | Hardware: {gpu_info}")
    print(f"==========================================================================")

    model = build_model(arch_name, num_classes=3, pretrained=False, freeze_vit_blocks=freeze_vit_flag).to(dev)
    total_params, trainable_params = count_parameters(model)
    params_M = round(total_params / 1e6, 2)
    trainable_M = round(trainable_params / 1e6, 2)

    res = {
        "arch": arch_name,
        "freeze_vit_blocks": freeze_vit_flag,
        "hardware": gpu_info,
        "params_M": params_M,
        "trainable_params_M": trainable_M,
        "batch_size": args.bs,
        "precision": "FP16 (torch.autocast)"
    }

    # 1. MACs (G) per image (1 x 3 x 224 x 224)
    model.eval()
    dummy_1 = torch.randn(1, 3, 224, 224, device=dev)
    try:
        from fvcore.nn import FlopCountAnalysis
        fca = FlopCountAnalysis(model, dummy_1)
        macs = fca.total()
        res["MACs_G"] = round(float(macs) / 1e9, 2)
        res["GMACs"] = res["MACs_G"]
        print(f"fvcore MACs: {res['MACs_G']} G")
    except Exception as e_fv:
        try:
            from thop import profile
            macs, _ = profile(model, inputs=(dummy_1,), verbose=False)
            res["MACs_G"] = round(float(macs) / 1e9, 2)
            res["GMACs"] = res["MACs_G"]
            print(f"thop MACs: {res['MACs_G']} G")
        except Exception as e_th:
            res["MACs_G"] = f"Error: {e_fv} / {e_th}"
            res["GMACs"] = res["MACs_G"]

    # 2. Training throughput + Peak VRAM
    model.train()
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)
    crit = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler(enabled=(dev == "cuda"))

    x_synth = torch.randn(args.bs, 3, 224, 224, device=dev)
    y_synth = torch.randint(0, 3, (args.bs,), device=dev)

    def train_step():
        opt.zero_grad(set_to_none=True)
        with torch.autocast(dev, dtype=torch.float16, enabled=(dev == "cuda")):
            out = model(x_synth)
            loss = crit(out, y_synth)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()

    # Warm-up
    for _ in range(args.warmup):
        train_step()

    if dev == "cuda":
        torch.cuda.reset_peak_memory_stats()

    throughputs = []
    for _ in range(args.repeats):
        if dev == "cuda":
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
            for _ in range(args.iters):
                train_step()
            end_event.record()
            torch.cuda.synchronize()
            elapsed_sec = start_event.elapsed_time(end_event) / 1000.0
        else:
            t0 = time.perf_counter()
            for _ in range(args.iters):
                train_step()
            elapsed_sec = time.perf_counter() - t0

        thr = (args.bs * args.iters) / elapsed_sec
        throughputs.append(thr)

    res["train_img_per_s"] = [round(float(np.mean(throughputs)), 1), round(float(np.std(throughputs)), 1)]
    res["train_epoch_s_9600_imgs_gpu_only"] = round(9600.0 / np.mean(throughputs), 1)

    if dev == "cuda":
        res["peak_mem_GB_train"] = round(float(torch.cuda.max_memory_allocated()) / 1e9, 2)

    # 3. Inference latency (bs=32 and bs=1)
    model.eval()
    for b_size in [args.bs, 1]:
        xb = torch.randn(b_size, 3, 224, 224, device=dev)
        # Warm-up
        with torch.no_grad(), torch.autocast(dev, dtype=torch.float16, enabled=(dev == "cuda")):
            for _ in range(args.warmup):
                model(xb)

        batch_latencies_ms = []
        for _ in range(args.repeats):
            if dev == "cuda":
                s_evt = torch.cuda.Event(enable_timing=True)
                e_evt = torch.cuda.Event(enable_timing=True)
                s_evt.record()
                with torch.no_grad(), torch.autocast(dev, dtype=torch.float16, enabled=(dev == "cuda")):
                    for _ in range(args.iters):
                        model(xb)
                e_evt.record()
                torch.cuda.synchronize()
                elapsed_ms = s_evt.elapsed_time(e_evt)
            else:
                t0 = time.perf_counter()
                with torch.no_grad():
                    for _ in range(args.iters):
                        model(xb)
                elapsed_ms = (time.perf_counter() - t0) * 1000.0

            batch_latencies_ms.append(elapsed_ms / args.iters)

        res[f"latency_ms_per_batch_bs{b_size}"] = [round(float(np.mean(batch_latencies_ms)), 3),
                                                  round(float(np.std(batch_latencies_ms)), 3)]
        res[f"latency_ms_per_image_bs{b_size}"] = round(float(np.mean(batch_latencies_ms)) / b_size, 3)

    out_file = args.out if args.out else f"exp3_{arch_name}.json"
    if os.path.dirname(out_file):
        os.makedirs(os.path.dirname(out_file), exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2)

    print(f"\n[OK] Exp 3 profiling saved to {out_file}:")
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
