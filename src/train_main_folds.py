"""Wrapper for training S1 (ResNet-50 Main) and S6 (ViT-S/16 Main) 5-fold CV.
Delegates to train_scenarios.py with --scenarios main.
"""
import sys
import subprocess

if __name__ == "__main__":
    cmd = [sys.executable, "train_scenarios.py", "--scenarios", "main"] + sys.argv[1:]
    print(f"[Delegating] {' '.join(cmd)}")
    sys.exit(subprocess.call(cmd))
