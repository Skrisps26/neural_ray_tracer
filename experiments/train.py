import gc

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset_loader import SubsetPixelDataset  # <--- CHANGED

# --- IMPORTS ---
from model import NeuralRenderer

# --- CONFIG ---
DATA_ROOT = "dataset_optimized"
BATCH_SIZE = 8192  # We can go back to high batch size because data is in RAM!
LR = 5e-4
EPOCHS = 100
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def train():
    print(f"--- Starting Subset Training on {DEVICE} ---")

    # 1. Model Setup (Created ONCE)
    model = NeuralRenderer(input_dim=81).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.L1Loss()

    # --- LOOP ---
    model.train()

    for epoch in range(EPOCHS):
        print(f"\n--- Epoch {epoch} ---")

        # 2. FRESH DATASET EVERY EPOCH
        # This picks a NEW random 25% of the room.
        # It takes ~10 seconds to load, but then training is INSTANT.
        dataset = SubsetPixelDataset(DATA_ROOT, subset_fraction=0.25)

        loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

        total_loss = 0
        num_batches = 0
        pbar = tqdm(loader)

        for batch in pbar:
            # 3. Move to GPU and Cast to Float32
            # Data was stored as Half (FP16) in RAM to save space.
            # We convert to Float (FP32) here for the math.
            pos = batch["pos"].to(DEVICE).float()
            albedo = batch["albedo"].to(DEVICE).float()
            normal = batch["normal"].to(DEVICE).float()
            target = batch["target"].to(DEVICE).float()

            optimizer.zero_grad()
            output = model(albedo, normal, pos)
            loss = criterion(torch.log1p(output), target)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

            if num_batches % 20 == 0:
                pbar.set_postfix({"Loss": f"{loss.item():.5f}"})

        avg_loss = total_loss / num_batches
        print(f">>> Avg Loss: {avg_loss:.6f}")

        # 4. Clean up RAM for next epoch
        del dataset
        del loader
        gc.collect()

        if epoch % 10 == 0:
            torch.save(model.state_dict(), f"model_epoch_{epoch}.pth")

    torch.save(model.state_dict(), "model_final.pth")
    print("--- Training Complete ---")


if __name__ == "__main__":
    train()
