import math
import time

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# --- CONFIG ---
RES_X, RES_Y = 640, 480
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LOOKAHEAD_FRAMES = 15.0  # How far into the future do we peek?
FUTURE_BUDGET = 0.30  # 25% of GPU power is spent on "Future Splats"

print(f"Running Predictive Neural Ray Tracer on: {DEVICE}")


# --- 1. MODEL (Hash Grid - Instant Memory) ---
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
        x_pos_norm = (x_pos + 2.0) / 4.0  # Normalize world space to [0,1]
        x_pos_norm = torch.clamp(x_pos_norm, 0.0, 1.0)
        embed = self.embedder(x_pos_norm)
        return self.net(torch.cat([embed, x_norm], dim=-1))


# --- 2. SCENE (Cornell Box with Physics) ---
class Scene:
    def __init__(self):
        # Walls
        self.planes = torch.tensor(
            [
                [1.0, 0.0, 0.0, 2.0],  # Left
                [-1.0, 0.0, 0.0, 2.0],  # Right
                [0.0, 1.0, 0.0, 2.0],  # Floor
                [0.0, -1.0, 0.0, 2.0],  # Ceiling
                [0.0, 0.0, 1.0, 2.0],  # Back
            ]
        ).to(DEVICE)

        self.plane_colors = torch.tensor(
            [
                [0.8, 0.1, 0.1],
                [0.1, 0.8, 0.1],
                [0.8, 0.8, 0.8],
                [0.8, 0.8, 0.8],
                [0.8, 0.8, 0.8],
            ]
        ).to(DEVICE)

        # Boxes
        self.boxes = [
            {
                "size": [0.6, 1.2, 0.6],
                "pos": [-0.6, -1.4, -0.6],
                "rot": 20,
                "color": [0.8, 0.8, 0.8],
            },
            {
                "size": [0.6, 0.6, 0.6],
                "pos": [0.6, -1.7, 0.3],
                "rot": -20,
                "color": [0.8, 0.8, 0.8],
            },
        ]

        for box in self.boxes:
            theta = math.radians(box["rot"])
            c, s = math.cos(theta), math.sin(theta)
            R = torch.tensor([[c, 0, s], [0, 1, 0], [-s, 0, c]], device=DEVICE)
            box["R"] = R
            box["invR"] = R.T

        self.light_pos = torch.tensor([0.0, 1.9, -1.0]).to(DEVICE)
        self.light_intensity = 8.0

    # (Same intersection logic as before, optimized for brevity)
    def intersect_box(self, ray_o, ray_d, box):
        # 1. Transform Ray to Box Local Space
        o_local = torch.matmul(
            ray_o - torch.tensor(box["pos"], device=DEVICE), box["R"]
        )
        d_local = torch.matmul(ray_d, box["R"])

        half = torch.tensor(box["size"], device=DEVICE) / 2.0
        inv_d = 1.0 / (d_local + 1e-6)

        # 2. Slab Method Intersection
        t0 = (-half - o_local) * inv_d
        t1 = (half - o_local) * inv_d

        # --- THE FIX ---
        # We are working with flattened rays [N, 3].
        # We just need to reduce across dimension 1 (the XYZ columns).

        # t_near = Max(Min(t0, t1).xyz)
        t_near = torch.max(torch.min(t0, t1), dim=1)[0]

        # t_far = Min(Max(t0, t1).xyz)
        t_far = torch.min(torch.max(t0, t1), dim=1)[0]
        # ----------------

        hit = (t_far > t_near) & (t_near > 0.001)

        # 3. Calculate Normal
        hit_p = o_local + d_local * t_near.unsqueeze(1)
        # Find which axis is closest to the box edge (that's the normal)
        axis = torch.argmax(torch.abs(hit_p) - half, dim=1)

        norm_local = torch.zeros_like(hit_p)
        norm_local.scatter_(
            1, axis.unsqueeze(1), torch.sign(hit_p.gather(1, axis.unsqueeze(1)))
        )

        return t_near, hit, torch.matmul(norm_local, box["invR"])

    def intersect(self, ray_o, ray_d):
        denom = torch.matmul(ray_d, self.planes[:, :3].T)
        denom = torch.where(denom.abs() < 1e-5, torch.ones_like(denom) * 1e-5, denom)
        t_planes = (
            -(torch.matmul(ray_o, self.planes[:, :3].T) + self.planes[:, 3]) / denom
        )
        t_room, idx = torch.min(
            torch.where(t_planes > 0.001, t_planes, torch.tensor(100.0, device=DEVICE)),
            dim=1,
        )

        t_final, mask_final = t_room, (t_room < 99.0)
        norm_final, col_final = self.planes[idx, :3], self.plane_colors[idx]

        for box in self.boxes:
            t_box, hit_box, norm_box = self.intersect_box(ray_o, ray_d, box)
            is_closer = hit_box & (t_box < t_final)
            t_final = torch.where(is_closer, t_box, t_final)
            mask_final = mask_final | hit_box
            norm_final = torch.where(is_closer.unsqueeze(1), norm_box, norm_final)
            col_final = torch.where(
                is_closer.unsqueeze(1),
                torch.tensor(box["color"], device=DEVICE),
                col_final,
            )

        return t_final, norm_final, col_final, mask_final


# --- 3. CAMERA & CONTROLS ---
# --- 3. CAMERA & CONTROLS (Updated with Rotation) ---
# --- 3. CAMERA (With Angular Velocity) ---
class Camera:
    def __init__(self):
        self.pos = torch.tensor([0.0, 0.0, 3.5], device=DEVICE)
        self.yaw = 0.0

        # Physics State
        self.velocity = torch.tensor([0.0, 0.0, 0.0], device=DEVICE)
        self.angular_velocity = 0.0

        # History for calculating deltas
        self.last_pos = self.pos.clone()
        self.last_yaw = 0.0

    def update(self, keys):
        # 1. Rotation Input
        rot_speed = 0.05
        if keys.get(ord("q")):
            self.yaw -= rot_speed
        if keys.get(ord("e")):
            self.yaw += rot_speed

        # 2. Movement Input
        c, s = math.cos(self.yaw), math.sin(self.yaw)
        forward = torch.tensor([s, 0, -c], device=DEVICE)
        right = torch.tensor([c, 0, s], device=DEVICE)

        speed = 0.1
        move_dir = torch.tensor([0.0, 0.0, 0.0], device=DEVICE)

        if keys.get(ord("w")):
            move_dir += forward
        if keys.get(ord("s")):
            move_dir -= forward
        if keys.get(ord("a")):
            move_dir -= right
        if keys.get(ord("d")):
            move_dir += right

        self.pos += move_dir * speed

        # 3. Calculate Physics Deltas (Linear & Angular)
        self.velocity = self.pos - self.last_pos
        self.angular_velocity = self.yaw - self.last_yaw

        # Update History
        self.last_pos = self.pos.clone()
        self.last_yaw = self.yaw

    # Updated to accept YAW override for prediction
    def get_rays(self, pos_override=None, yaw_override=None):
        origin = self.pos if pos_override is None else pos_override
        yaw = self.yaw if yaw_override is None else yaw_override

        y, x = torch.meshgrid(
            torch.linspace(1, -1, RES_Y, device=DEVICE),
            torch.linspace(-1, 1, RES_X, device=DEVICE),
            indexing="ij",
        )

        # 1. Base Rays (Forward -Z)
        rx, ry, rz = x, y, -torch.ones_like(x)

        # 2. Rotate Rays by Yaw (The override or the current)
        angle = torch.tensor(yaw, device=DEVICE)
        c, s = torch.cos(angle), torch.sin(angle)

        rot_x = rx * c - rz * s
        rot_z = rx * s + rz * c

        ray_d = torch.stack([rot_x, ry, rot_z], dim=-1)
        ray_d = ray_d / torch.norm(ray_d, dim=-1, keepdim=True)

        ray_o = origin.expand_as(ray_d).reshape(-1, 3)
        ray_d = ray_d.reshape(-1, 3)
        return ray_o, ray_d


# --- 4. RENDER LOOP (With Rotational Prediction) ---
def render_loop(scene, camera, model, optimizer):
    # --- A. RENDER CURRENT VIEW ---
    ray_o, ray_d = camera.get_rays()
    pos, normal, albedo, direct, target, valid = trace_and_shade(
        scene, model, ray_o, ray_d
    )
    pred_indirect = model(pos, normal)

    loss_current = nn.MSELoss()(pred_indirect[valid], target[valid])

    # --- B. ORACLE (FUTURE VIEW) ---
    is_moving = torch.norm(camera.velocity) > 0.001
    is_turning = abs(camera.angular_velocity) > 0.001

    # We will generate a debug image for the Oracle
    oracle_img = np.zeros((RES_Y, RES_X, 3), dtype=np.float32)

    if is_moving or is_turning:
        # 1. Calculate Future State
        # We clamp the rotation lookahead to avoid looking 360 degrees into the future
        future_yaw = camera.yaw + (
            camera.angular_velocity * 10.0
        )  # Look 10 frames ahead
        future_pos = camera.pos + (camera.velocity * 10.0)

        # 2. Trace Future Rays (Full Resolution for Debug)
        f_o, f_d = camera.get_rays(pos_override=future_pos, yaw_override=future_yaw)

        # We don't sub-sample here because we want to SEE the debug image
        # In a real game, you would sub-sample for speed.
        f_pos, f_norm, f_albedo, f_direct, f_target, f_valid = trace_and_shade(
            scene, model, f_o, f_d
        )
        f_pred = model(f_pos, f_norm)

        # 3. TURBO TRAIN (The Fix)
        # If we are turning fast, the network needs extra help.
        # We do a dedicated backward pass just for the future.
        loss_future = nn.MSELoss()(f_pred[f_valid], f_target[f_valid])

        # Combine losses? No. Let's do a weighted update.
        # We give the Future 5x more weight if we are turning fast.
        total_loss = loss_current + (loss_future * 5.0)

        # Generate Oracle Debug Image
        f_final = f_direct + (f_pred * f_albedo)
        oracle_img = f_final.reshape(RES_Y, RES_X, 3).detach().cpu().numpy()

    else:
        total_loss = loss_current

    # Optimization Step
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    # Tone Map Main Image
    final = direct + (pred_indirect * albedo)
    img = final.reshape(RES_Y, RES_X, 3).detach().cpu().numpy()

    # Common Tone Mapping for both
    def tonemap(im):
        im = im / (im + 1.0)
        return np.power(im, 1.0 / 2.2)

    return tonemap(img), tonemap(oracle_img), total_loss.item()


# --- 4. THE HYBRID TRAINER (Oracle Logic) ---
def trace_and_shade(scene, model, ray_o, ray_d):
    # 1. Primary Hit
    t, normal, albedo, mask = scene.intersect(ray_o, ray_d)
    pos = ray_o + ray_d * t.unsqueeze(1)

    # 2. Direct Light + Shadows
    L_dir = scene.light_pos - pos
    L_dist = torch.norm(L_dir, dim=1, keepdim=True)
    L_dir = L_dir / L_dist

    ndotl = torch.clamp(torch.sum(normal * L_dir, dim=1, keepdim=True), 0.0, 1.0)

    # Shadow Ray
    shadow_o = pos + normal * 0.01
    t_shadow, _, _, shadow_mask = scene.intersect(shadow_o, L_dir)

    # --- FIX 1: Use 1D Mask for Indexing ---
    # We squeeze L_dist to [N] to match t_shadow [N]
    in_shadow = shadow_mask & (t_shadow < L_dist.squeeze(1))

    direct = (scene.light_intensity / (L_dist**2 + 1.0)) * ndotl * albedo

    # ERROR WAS HERE: direct[in_shadow.unsqueeze(1)] -> Changed to direct[in_shadow]
    direct[in_shadow] = 0.0
    direct[~mask] = 0.0

    # 3. Bounce Ray (Bootstrap)
    rand = torch.randn_like(normal)
    rand = rand / torch.norm(rand, dim=1, keepdim=True)
    if_facing = torch.sum(rand * normal, dim=1, keepdim=True) < 0
    rand[if_facing.squeeze()] *= -1

    t_bounce, norm_B, col_B, mask_B = scene.intersect(pos + normal * 0.01, rand)
    pos_B = pos + rand * t_bounce.unsqueeze(1)

    with torch.no_grad():
        # Target Calculation at B
        L_dir_B = scene.light_pos - pos_B
        dist_B = torch.norm(L_dir_B, dim=1, keepdim=True)
        # Normalize Light Dir B
        L_dir_B_norm = L_dir_B / dist_B

        # Calculate Direct Light at B
        direct_B = (
            (scene.light_intensity / (dist_B**2 + 1.0))
            * torch.clamp(torch.sum(norm_B * L_dir_B_norm, 1, keepdim=True), 0, 1)
            * col_B
        )

        # Shadow Check at B
        t_s_B, _, _, s_m_B = scene.intersect(pos_B + norm_B * 0.01, L_dir_B_norm)

        # --- FIX 2: Apply 1D Mask Here Too ---
        is_shadow_B = s_m_B & (t_s_B < dist_B.squeeze(1))
        direct_B[is_shadow_B] = 0.0

        # Indirect at B
        indirect_B = model(pos_B, norm_B)
        target = (direct_B + indirect_B) * albedo

    return pos, normal, albedo, direct, target, mask & mask_B


# def render_loop(scene, camera, model, optimizer):
#     # A. Render Current View
#     ray_o, ray_d = camera.get_rays()
#     pos, normal, albedo, direct, target, valid = trace_and_shade(
#         scene, model, ray_o, ray_d
#     )
#     pred_indirect = model(pos, normal)

#     # B. The Oracle Step (Rotational + Linear)
#     # Check if we are moving OR turning
#     is_moving = torch.norm(camera.velocity) > 0.001
#     is_turning = abs(camera.angular_velocity) > 0.001

#     if is_moving or is_turning:
#         # 1. Extrapolate Position
#         future_pos = camera.pos + (camera.velocity * LOOKAHEAD_FRAMES)

#         # 2. Extrapolate Rotation (The Missing Piece!)
#         future_yaw = camera.yaw + (camera.angular_velocity * LOOKAHEAD_FRAMES)

#         # 3. Generate "Ghost" Rays
#         f_o, f_d = camera.get_rays(pos_override=future_pos, yaw_override=future_yaw)

#         # Sub-sample 25% for speed
#         indices = torch.randperm(f_o.shape[0])[: int(f_o.shape[0] * FUTURE_BUDGET)]
#         f_o_sub, f_d_sub = f_o[indices], f_d[indices]

#         # Train on Future
#         f_pos, f_norm, _, _, f_target, f_valid = trace_and_shade(
#             scene, model, f_o_sub, f_d_sub
#         )
#         f_pred = model(f_pos, f_norm)

#         loss_future = nn.MSELoss()(f_pred[f_valid], f_target[f_valid])
#         loss_current = nn.MSELoss()(pred_indirect[valid], target[valid])
#         total_loss = loss_current + loss_future
#     else:
#         total_loss = nn.MSELoss()(pred_indirect[valid], target[valid])

#     optimizer.zero_grad()
#     total_loss.backward()
#     optimizer.step()

#     final = direct + (pred_indirect * albedo)

#     # Tone Map
#     img = final.reshape(RES_Y, RES_X, 3).detach().cpu().numpy()
#     img = img / (img + 1.0)
#     img = np.power(img, 1.0 / 2.2)

#     return img, total_loss.item()


# --- 5. MAIN ---
# --- 5. MAIN (Updated Keys) ---
def main():
    scene = Scene()
    camera = Camera()
    model = HashNRC().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    print("--- DEBUG MODE ---")
    print("Small Window = What the AI is predicting (The Future)")

    while True:
        keys = {}
        key = cv2.waitKey(1) & 0xFF
        if key == ord("w"):
            keys[ord("w")] = True
        if key == ord("s"):
            keys[ord("s")] = True
        if key == ord("a"):
            keys[ord("a")] = True
        if key == ord("d"):
            keys[ord("d")] = True
        if key == ord("q"):
            keys[ord("q")] = True
        if key == ord("e"):
            keys[ord("e")] = True
        if key == 27:
            break

        camera.update(keys)

        # Get BOTH images
        img, oracle_view, loss = render_loop(scene, camera, model, optimizer)

        # Create UI
        ui = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        # --- PICTURE IN PICTURE ---
        # Resize Oracle View to 1/4 size
        thumb_h, thumb_w = RES_Y // 3, RES_X // 3
        thumb = cv2.resize(
            cv2.cvtColor(oracle_view, cv2.COLOR_RGB2BGR), (thumb_w, thumb_h)
        )

        # Overlay border
        cv2.rectangle(thumb, (0, 0), (thumb_w - 1, thumb_h - 1), (0, 255, 0), 1)

        # Paste into Bottom Right corner
        y_offset = RES_Y - thumb_h - 10
        x_offset = RES_X - thumb_w - 10
        ui[y_offset : y_offset + thumb_h, x_offset : x_offset + thumb_w] = thumb

        # Status Text
        lin_s = torch.norm(camera.velocity).item()
        rot_s = abs(camera.angular_velocity)
        active = (lin_s > 0.001) or (rot_s > 0.001)

        cv2.putText(
            ui,
            f"Oracle: {'ACTIVE' if active else 'SLEEP'}",
            (x_offset, y_offset - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (0, 255, 0),
            1,
        )

        cv2.imshow("Debug View", ui)


if __name__ == "__main__":
    main()
