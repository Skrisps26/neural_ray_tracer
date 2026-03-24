import math

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# --- CONFIG ---
RES_X, RES_Y = 640, 480  # Lower res for speed on 4GB GPU
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Running Cornell Box Ray Tracer on: {DEVICE}")


# --- 1. MODEL (Hash Grid) ---
class HashEmbedder(nn.Module):
    def __init__(self, num_levels=12, base_res=16, max_res=512, log2_hashmap_size=15):
        super().__init__()
        self.num_levels = num_levels
        self.hashmap_size = 2**log2_hashmap_size
        b = np.exp((np.log(max_res) - np.log(base_res)) / (num_levels - 1))
        self.resolutions = [int(base_res * b**i) for i in range(num_levels)]

        self.embeddings = nn.ParameterList(
            [
                nn.Parameter(
                    torch.FloatTensor(self.hashmap_size, 2).uniform_(-1e-4, 1e-4)
                )
                for _ in range(num_levels)
            ]
        )

    def forward(self, x):
        outputs = []
        for i, res in enumerate(self.resolutions):
            embed = self.embeddings[i]
            scaled_x = x * res
            x0 = torch.floor(scaled_x).long()
            primes = [1, 2654435761, 805459861]
            p = x0 * torch.tensor(primes, device=x.device)
            h = (p[:, 0] ^ p[:, 1] ^ p[:, 2]) % self.hashmap_size
            outputs.append(embed[h])
        return torch.cat(outputs, dim=-1)


class HashNRC(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedder = HashEmbedder()
        self.net = nn.Sequential(
            nn.Linear(27, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 3),
            nn.Softplus(),
        )

    def forward(self, x_pos, x_norm):
        x_pos_norm = (x_pos + 2.0) / 4.0
        x_pos_norm = torch.clamp(x_pos_norm, 0.0, 1.0)
        embed = self.embedder(x_pos_norm)
        return self.net(torch.cat([embed, x_norm], dim=-1))


# --- 2. SCENE (With Boxes) ---
class Scene:
    def __init__(self):
        # Walls: Left(Red), Right(Green), Floor(White), Ceiling(White), Back(White)
        self.planes = torch.tensor(
            [
                [1.0, 0.0, 0.0, 2.0],
                [-1.0, 0.0, 0.0, 2.0],
                [0.0, 1.0, 0.0, 2.0],
                [0.0, -1.0, 0.0, 2.0],
                [0.0, 0.0, 1.0, 2.0],
            ]
        ).to(DEVICE)

        self.plane_colors = torch.tensor(
            [
                [0.8, 0.1, 0.1],  # Red
                [0.1, 0.8, 0.1],  # Green
                [0.8, 0.8, 0.8],  # White
                [0.8, 0.8, 0.8],  # White
                [0.8, 0.8, 0.8],  # White
            ]
        ).to(DEVICE)

        # Boxes: Size, Pos, Rot(Degrees), Color
        self.boxes = [
            {
                "size": [0.6, 1.2, 0.6],
                "pos": [-0.6, -1.4, -0.6],
                "rot": 20,
                "color": [0.8, 0.8, 0.8],
            },  # Tall Left
            {
                "size": [0.6, 0.6, 0.6],
                "pos": [0.6, -1.7, 0.3],
                "rot": -20,
                "color": [0.8, 0.8, 0.8],
            },  # Short Right
        ]

        # Precompute Box Transforms
        for box in self.boxes:
            theta = math.radians(box["rot"])
            c, s = math.cos(theta), math.sin(theta)
            # Rotation Matrix (Y-axis)
            R = torch.tensor([[c, 0, s], [0, 1, 0], [-s, 0, c]], device=DEVICE)
            box["R"] = R
            box["invR"] = R.T

        self.light_pos = torch.tensor([0.0, 1.9, -1.0]).to(DEVICE)
        self.light_intensity = 8.0

    def intersect_box(self, ray_o, ray_d, box):
        # Transform Ray to Box Local Space
        o_local = torch.matmul(
            ray_o - torch.tensor(box["pos"], device=DEVICE), box["R"]
        )
        d_local = torch.matmul(ray_d, box["R"])

        half = torch.tensor(box["size"], device=DEVICE) / 2.0
        inv_d = 1.0 / (d_local + 1e-6)

        t0 = (-half - o_local) * inv_d
        t1 = (half - o_local) * inv_d

        t_min = torch.min(t0, t1)
        t_max = torch.max(t0, t1)

        t_near = torch.max(torch.max(t_min[..., 0], t_min[..., 1]), t_min[..., 2])
        t_far = torch.min(torch.min(t_max[..., 0], t_max[..., 1]), t_max[..., 2])

        hit = (t_far > t_near) & (t_near > 0.001)

        # Calculate Normal (Heuristic: which face is hit point closest to?)
        hit_p = o_local + d_local * t_near.unsqueeze(1)
        dist = torch.abs(hit_p) - half
        axis = torch.argmax(dist, dim=1)

        norm_local = torch.zeros_like(hit_p)
        norm_local.scatter_(
            1, axis.unsqueeze(1), torch.sign(hit_p.gather(1, axis.unsqueeze(1)))
        )
        norm_world = torch.matmul(norm_local, box["invR"])  # Transform normal back

        return t_near, hit, norm_world

    def intersect(self, ray_o, ray_d):
        # 1. Plane Intersection
        denom = torch.matmul(ray_d, self.planes[:, :3].T)
        denom = torch.where(denom.abs() < 1e-5, torch.ones_like(denom) * 1e-5, denom)
        t_planes = (
            -(torch.matmul(ray_o, self.planes[:, :3].T) + self.planes[:, 3]) / denom
        )

        mask_planes = t_planes > 0.001
        t_room, idx_room = torch.min(
            torch.where(mask_planes, t_planes, torch.tensor(100.0, device=DEVICE)),
            dim=1,
        )

        # Init Final Values
        t_final = t_room
        mask_final = t_room < 99.0
        norm_final = self.planes[idx_room, :3]
        col_final = self.plane_colors[idx_room]

        # 2. Box Intersection
        for box in self.boxes:
            t_box, hit_box, norm_box = self.intersect_box(ray_o, ray_d, box)

            # Update if hit and closer
            is_closer = hit_box & (t_box < t_final)

            t_final = torch.where(is_closer, t_box, t_final)
            mask_final = mask_final | hit_box

            # Update Properties
            norm_final = torch.where(is_closer.unsqueeze(1), norm_box, norm_final)
            box_col = torch.tensor(box["color"], device=DEVICE).expand_as(col_final)
            col_final = torch.where(is_closer.unsqueeze(1), box_col, col_final)

        return t_final, norm_final, col_final, mask_final


# --- 3. RENDER LOOP (Updated for Boxes) ---
def render_frame(scene, model, optimizer):
    # A. PRIMARY RAYS
    y, x = torch.meshgrid(
        torch.linspace(1, -1, RES_Y, device=DEVICE),
        torch.linspace(-1, 1, RES_X, device=DEVICE),
        indexing="ij",
    )
    ray_d = torch.stack([x, y, -torch.ones_like(x)], dim=-1)
    ray_d = ray_d / torch.norm(ray_d, dim=-1, keepdim=True)
    ray_o = torch.tensor([0.0, 0.0, 3.5], device=DEVICE).expand_as(ray_d).reshape(-1, 3)
    ray_d = ray_d.reshape(-1, 3)

    # B. G-BUFFER (Call new intersect)
    t, normal_A, albedo_A, mask = scene.intersect(ray_o, ray_d)
    pos_A = ray_o + ray_d * t.unsqueeze(1)

    # Zero out background
    pos_A[~mask] = 0
    normal_A[~mask] = 0
    albedo_A[~mask] = 0

    # C. DIRECT LIGHTING
    L_dir = scene.light_pos - pos_A
    dist_sq = torch.sum(L_dir**2, dim=1, keepdim=True)
    L_dist = torch.sqrt(dist_sq)
    L_dir = L_dir / L_dist

    ndotl = torch.clamp(torch.sum(normal_A * L_dir, dim=1, keepdim=True), 0.0, 1.0)

    # Shadow Ray
    shadow_o = pos_A + normal_A * 0.01
    t_shadow, _, _, shadow_mask = scene.intersect(shadow_o, L_dir)

    # Shadow Logic (1D Mask Fix)
    is_shadow = shadow_mask & (t_shadow < L_dist.squeeze(1))

    direct_light = (scene.light_intensity / (dist_sq + 1.0)) * ndotl * albedo_A
    direct_light[is_shadow] = 0.0
    direct_light[~mask] = 0.0

    # D. INDIRECT LIGHTING (Bootstrap)
    pred_indirect_A = model(pos_A, normal_A)

    # Bounce Ray
    rand = torch.randn_like(normal_A)
    rand = rand / torch.norm(rand, dim=1, keepdim=True)
    if_facing = torch.sum(rand * normal_A, dim=1, keepdim=True) < 0
    rand[if_facing.squeeze()] *= -1

    t_bounce, norm_B, col_B, mask_B = scene.intersect(pos_A + normal_A * 0.01, rand)
    pos_B = pos_A + rand * t_bounce.unsqueeze(1)

    # Target Calculation
    with torch.no_grad():
        # Direct at B
        L_dir_B = scene.light_pos - pos_B
        dist_sq_B = torch.sum(L_dir_B**2, dim=1, keepdim=True)
        L_dist_B = torch.sqrt(dist_sq_B)
        L_dir_B = L_dir_B / L_dist_B
        ndotl_B = torch.clamp(
            torch.sum(norm_B * L_dir_B, dim=1, keepdim=True), 0.0, 1.0
        )

        # Shadow at B
        t_shadow_B, _, _, shadow_mask_B = scene.intersect(
            pos_B + norm_B * 0.01, L_dir_B
        )
        is_shadow_B = shadow_mask_B & (t_shadow_B < L_dist_B.squeeze(1))

        direct_B = (scene.light_intensity / (dist_sq_B + 1.0)) * ndotl_B * col_B
        direct_B[is_shadow_B] = 0.0

        # Indirect at B
        indirect_B = model(pos_B, norm_B)

        # Rendering Equation
        incoming_B = direct_B + indirect_B
        target_A = incoming_B * albedo_A

    # E. TRAIN
    valid_rays = mask & mask_B
    loss = nn.MSELoss()(pred_indirect_A[valid_rays], target_A[valid_rays])

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # F. COMPOSITE & TONE MAP
    final_color = direct_light + (pred_indirect_A * albedo_A)

    color = final_color.reshape(RES_Y, RES_X, 3).detach().cpu().numpy()
    # ACES Tone Map
    a, b, c, d, e = 2.51, 0.03, 2.43, 0.59, 0.14
    color = (color * (a * color + b)) / (color * (c * color + d) + e)
    color = np.clip(color, 0.0, 1.0)
    color = color ** (1 / 2.2)  # Gamma

    return color, loss.item()


# --- 4. MAIN ---
def main():
    scene = Scene()
    model = HashNRC().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=0.01, betas=(0.9, 0.99))

    print("Ray Tracer Active. [Q] to Quit.")

    frame = 0
    while True:
        img, loss = render_frame(scene, model, optimizer)

        cv2.imshow(
            "Cornell Box - Neural Ray Tracer", cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        )
        print(f"Frame {frame} | Loss: {loss:.4f}", end="\r")

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        frame += 1


if __name__ == "__main__":
    main()
