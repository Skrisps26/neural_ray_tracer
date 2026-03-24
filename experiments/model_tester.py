import torch
import torch.nn as nn
import torch.optim as optim

from model import NeuralRenderer

# CONFIG
CHUNK_PATH = "dataset_optimized/kitchen_chunk_00.pt"  # Pick ANY valid chunk file
LR = 1e-4
EPOCHS = 500  # We will hammer this single chunk
DEVICE = "cuda"


def train_overfit():
    print(f"--- OVERFITTING TEST on {DEVICE} ---")

    # 1. Load ONE Chunk
    data = torch.load(CHUNK_PATH, map_location=DEVICE)
    # Extract & Scale (Use NEW scale 150.0)
    SCENE_SCALE = 150.0

    pos = (data["pos"].float() - data["cam_pos"].float()) / SCENE_SCALE
    # Simple masking
    mask = (pos.norm(dim=1, keepdim=True) < 1.5).float()
    pos = pos * mask

    albedo = data["albedo"].float()
    normal = data["normal"].float()

    # Target: Log Irradiance
    target = torch.log1p(torch.clamp(data["beauty"].float() / (albedo + 0.01), 0, 100))

    # 2. Model
    model = NeuralRenderer(input_dim=81).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.MSELoss()  # Strict loss

    model.train()

    for epoch in range(EPOCHS):
        optimizer.zero_grad()
        output = model(albedo, normal, pos)
        loss = criterion(torch.log1p(torch.clamp(output, 0, 1000)), target)
        loss.backward()
        optimizer.step()

        if epoch % 50 == 0:
            print(f"Epoch {epoch} | Loss: {loss.item():.6f}")

    # 3. Save
    torch.save(model.state_dict(), "model_overfit.pth")
    print(
        "Test Complete. Run inference using 'model_overfit.pth' and SCENE_SCALE=150.0"
    )


if __name__ == "__main__":
    train_overfit()
