import math

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# --- CONFIG ---
RES_X, RES_Y = 320, 240
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LOOKAHEAD = 5.0
TRAIN_BATCH = 1024

print(f"Running NEURAL GAUSSIAN LUMEN on: {DEVICE}")


# --- 1. THE BRAIN (Neural Radiance Cache) ---
# Replaces the explicit "Probe Grid".
# It learns the light color at any position (x,y,z).
class NeuralCache(nn.Module):
    def __init__(self):
        super().__init__()
        # Standard hash grid for memory
        self.embedder = nn.Embedding(4096, 16)
        self.net = nn.Sequential(
            nn.Linear(16 + 3, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 3),
            nn.Sigmoid(),  # Color is 0.0-1.0
        )

    def forward(self, pos, normal):
        # 1. Hashing (Spatial Memory)
        # Simple spatial hash for demo speed
        scaled = (pos * 10).long()
        idx = (
            scaled[:, 0] * 73856093 ^ scaled[:, 1] * 19349663 ^ scaled[:, 2] * 83492791
        ) % 4096
        feat = self.embedder(idx)

        # 2. Inference
        return self.net(torch.cat([feat, normal], dim=-1))


# --- 2. THE BODY (Gaussian Scene) ---
# Explicit Geometry. We utilize this for collisions/intersections.
class GaussianScene:
    def __init__(self):
        self.pos = torch.zeros(0, 3).to(DEVICE)
        self.col = torch.zeros(0, 3).to(DEVICE)
        self.vel = torch.zeros(0, 3).to(DEVICE)

    def add_blob(self, center, color, points=500, dynamic=False):
        # Create a cluster of points
        new_pos = torch.randn(points, 3).to(DEVICE) * 0.1 + torch.tensor(center).to(
            DEVICE
        )
        new_col = torch.tensor(color).to(DEVICE).repeat(points, 1)
        new_vel = torch.zeros(points, 3).to(DEVICE)

        start_idx = len(self.pos)
        self.pos = torch.cat([self.pos, new_pos])
        self.col = torch.cat([self.col, new_col])
        self.vel = torch.cat([self.vel, new_vel])
        return torch.arange(start_idx, start_idx + points)

    def update(self, keys, idx):
        # Physics Logic
        move = torch.tensor([0.0, 0.0, 0.0]).to(DEVICE)
        if keys.get(ord("i")):
            move[2] += 0.1
        if keys.get(ord("k")):
            move[2] -= 0.1
        if keys.get(ord("j")):
            move[0] -= 0.1
        if keys.get(ord("l")):
            move[0] += 0.1

        self.vel *= 0.9  # Friction
        self.vel[idx] += move * 0.05
        self.pos[idx] += self.vel[idx]

    def get_future_pos(self):
        # THE LOOKAHEAD: Where will the scene be in 5 frames?
        return self.pos + (self.vel * LOOKAHEAD)


# --- 3. THE ENGINE ---
def render_and_train(scene, model, opt, cam_pos, cam_yaw):
    # A. RAY MARCHING (Geometry Pass)
    # We shoot rays. Instead of triangles, we find the nearest Gaussian.
    y, x = torch.meshgrid(
        torch.linspace(1, -1, RES_Y).to(DEVICE),
        torch.linspace(-1, 1, RES_X).to(DEVICE),
        indexing="ij",
    )
    rx, ry, rz = x, y, -torch.ones_like(x)
    angle = torch.tensor(cam_yaw).to(DEVICE)
    c, s = torch.cos(angle), torch.sin(angle)
    rd = torch.stack([rx * c - rz * s, ry, rx * s + rz * c], -1)
    rd = rd / rd.norm(dim=-1, keepdim=True)
    ro = cam_pos.expand_as(rd)

    # Cheat Intersection: Find closest Gaussian to ray tip (at distance 3.0)
    # Real implementation would use Rasterization or Sphere Tracing.
    ray_tips = ro + rd * 3.0

    # Brute Force Search (Slow in Python, fast in C++/Shader)
    # We downsample for speed in this Python demo
    view_subset = torch.randint(0, RES_X * RES_Y, (TRAIN_BATCH,)).to(DEVICE)
    flat_tips = ray_tips.reshape(-1, 3)[view_subset]

    # B. NEURAL TRAINING (Lighting Pass)
    # 1. Look into the Future (use lookahead positions)
    future_geo = scene.get_future_pos()

    # 2. Find nearest Gaussian (Collision Detection)
    dists = torch.cdist(flat_tips, future_geo)  # [Batch, NumGaussians]
    min_dist, hit_idx = dists.min(dim=1)

    # 3. Get "True" Color from Geometry
    hit_mask = min_dist < 0.2  # Did we hit a glowing blob?
    target_color = scene.col[hit_idx]

    # 4. Train Neural Net
    # "Learn that at Position P, the light is Color C"
    # Even if we haven't rendered it yet, the Lookahead tells us it will be there.
    if hit_mask.any():
        valid_tips = flat_tips[hit_mask]
        valid_targets = target_color[hit_mask]

        # Fake Normal (pointing to camera)
        valid_norms = -rd.reshape(-1, 3)[view_subset][hit_mask]

        pred_color = model(valid_tips, valid_norms)
        loss = ((pred_color - valid_targets) ** 2).mean()

        opt.zero_grad()
        loss.backward()
        opt.step()

    # C. INFERENCE (Render to Screen)
    # We just query the Neural Network for every pixel!
    # The NN has learned the scene color from the training step above.
    with torch.no_grad():
        # Downsample visualization 4x for speed
        vis_tips = ray_tips[::4, ::4].reshape(-1, 3)
        vis_norms = -rd[::4, ::4].reshape(-1, 3)

        pixel_colors = model(vis_tips, vis_norms)
        img = pixel_colors.reshape(RES_Y // 4, RES_X // 4, 3).cpu().numpy()
        img = cv2.resize(img, (RES_X, RES_Y), interpolation=cv2.INTER_NEAREST)

    return img


# --- 4. MAIN ---
def main():
    scene = GaussianScene()
    model = NeuralCache().to(DEVICE)
    opt = optim.Adam(model.parameters(), lr=0.01)

    # Static Red/Green Walls
    scene.add_blob([-1.0, 0, 3], [1.0, 0.0, 0.0])
    scene.add_blob([1.0, 0, 3], [0.0, 1.0, 0.0])

    # Dynamic Blue Player
    p_idx = scene.add_blob([0, 0, 3], [0.0, 0.5, 1.0], dynamic=True)

    cam_pos = torch.tensor([0.0, 0.0, 0.0]).to(DEVICE)
    cam_yaw = 0.0

    print("Neural + Gaussian Hybrid.")
    print("I/J/K/L to move the Blue Blob.")

    while True:
        keys = {}
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break
        if key == ord("i"):
            keys[ord("i")] = True
        if key == ord("k"):
            keys[ord("k")] = True
        if key == ord("j"):
            keys[ord("j")] = True
        if key == ord("l"):
            keys[ord("l")] = True

        scene.update(keys, p_idx)
        img = render_and_train(scene, model, opt, cam_pos, cam_yaw)
        cv2.imshow("Neural Gaussian", img)


if __name__ == "__main__":
    main()
