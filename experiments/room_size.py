import glob
import os

import numpy as np
import torch
from tqdm import tqdm

DATA_ROOT = "dataset_optimized"


def check_bounds():
    files = sorted(glob.glob(os.path.join(DATA_ROOT, "*.pt")))
    if not files:
        print("No files found!")
        return

    global_max = 0.0

    print("Scanning dataset for coordinate bounds...")
    for f in tqdm(files):
        data = torch.load(f)
        pos = data["pos"]  # (B, 3, H, W)

        # We need to ignore "Infinite" background points
        # Let's assume anything > 1000 is background for now
        mask = pos.abs() < 1000
        if mask.sum() == 0:
            continue

        local_max = pos[mask].abs().max().item()
        if local_max > global_max:
            global_max = local_max

    print(f"\n>>> YOUR SCENE SCALE IS: {global_max:.2f}")
    print(f">>> You should divide positions by approx: {global_max * 1.2:.2f}")


if __name__ == "__main__":
    check_bounds()
