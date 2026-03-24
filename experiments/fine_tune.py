import gc
import glob
import os
import random

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

# --- IMPORTS ---
from model import NeuralRenderer

# --- CONFIG ---
DATA_ROOT = "dataset_optimized"
# FIND THE LATEST CHECKPOINT (Adjust number if needed)
START_CHECKPOINT = "model_epoch_99.pth"

BATCH_SIZE = 8192
STEPS_PER_CHUNK = 50
LR = 5e-5  # <--- 10x SMALLER (Precision Mode)
EPOCHS = 50  # Run 50 more epochs
DEVICE = "cuda"


def load_and_process_chunk_to_vram(f_path):
    # (Same loader as before)
    try:
        data = torch.load(f_path, map_location="cpu")
        pos_raw = torch.nan_to_num(data["pos"], nan=0.0)
        cam_pos = torch.nan_to_num(data["cam_pos"], nan=0.0)
        beauty = torch.nan_to_num(data["beauty"], nan=0.0)
        albedo = torch.nan_to_num(data["albedo"], nan=0.0)
        normal = torch.nan_to_num(data["normal"], nan=0.0)

        scene_scale = 150
        pos_relative = pos_raw - cam_pos
        dist = torch.norm(pos_relative, dim=1, keepdim=True)
        mask = (dist < (scene_scale * 1.5)).float()
        pos_final = (pos_relative * mask) / scene_scale

        epsilon = 0.01
        target_irradiance = beauty / (albedo + epsilon)
        target_final = torch.log1p(torch.clamp(target_irradiance, 0, 100.0))

        return {
            "pos": pos_final.permute(0, 2, 3, 1)
            .reshape(-1, 3)
            .to(DEVICE, dtype=torch.float32),
            "albedo": albedo.permute(0, 2, 3, 1)
            .reshape(-1, 3)
            .to(DEVICE, dtype=torch.float32),
            "normal": normal.permute(0, 2, 3, 1)
            .reshape(-1, 3)
            .to(DEVICE, dtype=torch.float32),
            "target": target_final.permute(0, 2, 3, 1)
            .reshape(-1, 3)
            .to(DEVICE, dtype=torch.float32),
        }
    except Exception:
        return None


def train():
    print(f"--- Starting Fine-Tuning (LR={LR}) on {DEVICE} ---")

    all_files = sorted(glob.glob(os.path.join(DATA_ROOT, "*.pt")))

    # 1. Load Model Structure
    model = NeuralRenderer(input_dim=81).to(DEVICE)

    # 2. Load Weights from Epoch 50
    if os.path.exists(START_CHECKPOINT):
        print(f">>> Loading weights from {START_CHECKPOINT}...")
        model.load_state_dict(torch.load(START_CHECKPOINT))
    else:
        print(f"!!! CRITICAL: Could not find {START_CHECKPOINT}. Check filename.")
        return

    # 3. New Optimizer with Low LR
    optimizer = optim.Adam(model.parameters(), lr=LR)

    # 4. SWAPPING TO MSE LOSS (Aggressive on highlights)
    criterion = nn.MSELoss()

    model.train()

    for epoch in range(51, 51 + EPOCHS):
        print(f"\n--- Fine-Tune Epoch {epoch} ---")
        random.shuffle(all_files)

        total_loss = 0
        chunks_processed = 0
        pbar = tqdm(all_files, desc=f"Ep {epoch}")

        for f_path in pbar:
            chunk_data = load_and_process_chunk_to_vram(f_path)
            if chunk_data is None:
                continue

            c_pos = chunk_data["pos"]
            c_albedo = chunk_data["albedo"]
            c_normal = chunk_data["normal"]
            c_target = chunk_data["target"]

            if c_pos.shape[0] < BATCH_SIZE:
                continue

            chunk_loss = 0
            for _ in range(STEPS_PER_CHUNK):
                idx = torch.randint(0, c_pos.shape[0], (BATCH_SIZE,), device=DEVICE)

                optimizer.zero_grad()
                output = model(c_albedo[idx], c_normal[idx], c_pos[idx])

                # Clamp for safety
                output_safe = torch.clamp(output, min=0.0, max=1000.0)

                loss = criterion(torch.log1p(output_safe), c_target[idx])

                if torch.isnan(loss):
                    continue

                loss.backward()
                optimizer.step()
                chunk_loss += loss.item()

            del chunk_data, c_pos, c_albedo, c_normal, c_target

            total_loss += chunk_loss / STEPS_PER_CHUNK
            chunks_processed += 1
            pbar.set_postfix({"MSE Loss": f"{chunk_loss / STEPS_PER_CHUNK:.5f}"})

        avg_loss = total_loss / max(1, chunks_processed)
        print(f">>> Epoch {epoch} Done. Avg MSE: {avg_loss:.6f}")
        torch.save(model.state_dict(), f"model_epoch_{epoch}.pth")

        gc.collect()
        torch.cuda.empty_cache()

    torch.save(model.state_dict(), "model_final_finetuned.pth")


if __name__ == "__main__":
    train()
