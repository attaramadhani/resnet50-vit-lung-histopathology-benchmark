"""Pre-processes and caches LC25000 images to 256 px to eliminate disk I/O bottlenecks.
Section 1 of implementation_plan2.md.
"""
import os
import argparse
import pandas as pd
from PIL import Image
from concurrent.futures import ThreadPoolExecutor
from common import get_lc25000_root

def resize_image(args_tuple):
    src_path, dst_path, target_size = args_tuple
    if os.path.exists(dst_path):
        return True
    try:
        os.makedirs(os.path.dirname(dst_path), exist_ok=True)
        with Image.open(src_path) as im:
            im = im.convert("RGB")
            # LC25000 images are 768x768 square; resize to 256x256
            im_resized = im.resize((target_size, target_size), Image.Resampling.BILINEAR)
            im_resized.save(dst_path, quality=95)
        return True
    except Exception as e:
        print(f"Error resizing {src_path}: {e}")
        return False

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", "--raw-root", "--raw_root", dest="data_root", default="")
    ap.add_argument("--csv", default="lc25000_lung_all.csv")
    ap.add_argument("--out_dir", "--cache-root", "--cache_root", dest="out_dir", default="data/cache_lc25000_256")
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    root = get_lc25000_root(args.data_root)
    print(f"[Cache 256] Source LC25000 root: {root}")
    print(f"[Cache 256] Target cache dir   : {args.out_dir}")

    if not os.path.exists(args.csv):
        alt_csv = os.path.join("data_splits", os.path.basename(args.csv))
        if os.path.exists(alt_csv):
            args.csv = alt_csv
        else:
            raise FileNotFoundError(f"Cannot find CSV at {args.csv}")

    df = pd.read_csv(args.csv)
    tasks = []
    for _, r in df.iterrows():
        rel_p = str(r["path"]).lstrip("/\\")
        src_p = os.path.join(root, rel_p)
        dst_p = os.path.join(args.out_dir, rel_p)
        tasks.append((src_p, dst_p, args.size))

    print(f"[Cache 256] Resizing {len(tasks)} images to {args.size}x{args.size} using {args.workers} workers...")
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(resize_image, tasks))

    success = sum(results)
    print(f"[Cache 256] Finished! Successfully cached {success}/{len(tasks)} images to {args.out_dir}")

    # Write cache path to cache_root.txt for automatic pickup
    with open("cache_root.txt", "w") as f:
        f.write(os.path.abspath(args.out_dir) + "/")

if __name__ == "__main__":
    main()
