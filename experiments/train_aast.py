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
BATCH_SIZE = 8192
STEPS_PER_CHUNK = 50
LR = 1e-4  # <--- Lowered for stability
EPOCHS = 100
DEVICE = "cuda"


def load_and_process_chunk_to_vram(f_path):
    try:
        data = torch.load(f_path, map_location="cpu")

        # Extract
        pos_raw = torch.nan_to_num(data["pos"], nan=0.0)
        cam_pos = torch.nan_to_num(data["cam_pos"], nan=0.0)
        beauty = torch.nan_to_num(data["beauty"], nan=0.0)
        albedo = torch.nan_to_num(data["albedo"], nan=0.0)
        normal = torch.nan_to_num(data["normal"], nan=0.0)

        # Process
        scene_scale = 150
        pos_relative = pos_raw - cam_pos
        dist = torch.norm(pos_relative, dim=1, keepdim=True)
        mask = (dist < (scene_scale * 1.5)).float()
        pos_final = (pos_relative * mask) / scene_scale

        epsilon = 0.01
        target_irradiance = beauty / (albedo + epsilon)

        # Clamp Target excessively to prevent FP16 overflow (just in case)
        target_final = torch.log1p(torch.clamp(target_irradiance, 0, 100.0))

        # Move to VRAM as FLOAT32 (Stable)
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
    except Exception as e:
        # print(f"Error loading {f_path}: {e}")
        return None


def train():
    print(f"--- Starting Stable Training on {DEVICE} ---")

    all_files = sorted(glob.glob(os.path.join(DATA_ROOT, "*.pt")))
    if not all_files:
        print("Error: No files found.")
        return

    # Model Setup
    model = NeuralRenderer(input_dim=81).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.L1Loss()

    # We REMOVED the Scaler (No Mixed Precision)

    model.train()

    for epoch in range(EPOCHS):
        print(f"\n--- Epoch {epoch} ---")
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

            num_pixels = c_pos.shape[0]
            if num_pixels < BATCH_SIZE:
                continue

            chunk_loss = 0

            for _ in range(STEPS_PER_CHUNK):
                # 1. Random Indices
                idx = torch.randint(0, num_pixels, (BATCH_SIZE,), device=DEVICE)

                b_pos = c_pos[idx]
                b_albedo = c_albedo[idx]
                b_normal = c_normal[idx]
                b_target = c_target[idx]

                optimizer.zero_grad()

                # 2. Forward Pass (No Autocast)
                output = model(b_albedo, b_normal, b_pos)

                # --- SAFETY CLAMP ---
                # Force prediction to be valid before Log
                # This prevents Inf/NaN if the model initializes poorly
                output_safe = torch.clamp(output, min=0.0, max=1000.0)

                loss = criterion(torch.log1p(output_safe), b_target)

                # 3. NaN Panic Check
                if torch.isnan(loss):
                    print("!!! NaN Detected. Skipping Step.")
                    optimizer.zero_grad()
                    continue

                loss.backward()

                # 4. Gradient Clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

                optimizer.step()

                chunk_loss += loss.item()

            # Cleanup
            del (
                chunk_data,
                c_pos,
                c_albedo,
                c_normal,
                c_target,
                idx,
                b_pos,
                output,
                loss,
            )

            total_loss += chunk_loss / STEPS_PER_CHUNK
            chunks_processed += 1

            pbar.set_postfix({"Loss": f"{chunk_loss / STEPS_PER_CHUNK:.5f}"})

        # End Epoch
        if chunks_processed > 0:
            avg_loss = total_loss / chunks_processed
            print(f">>> Epoch {epoch} Done. Avg Loss: {avg_loss:.6f}")
            torch.save(model.state_dict(), f"model_epoch_{epoch}.pth")

        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    train()
