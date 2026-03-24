import math
import time

import numpy as np
import taichi as ti
import torch
import torch.nn as nn
import torch.optim as optim

# ============================================================================
# CONFIGURATION
# ============================================================================
ti.init(arch=ti.cuda, device_memory_fraction=0.9, unrolling_limit=0)

RES_X, RES_Y = 640, 480
INTERNAL_X, INTERNAL_Y = 640, 360
MAX_SPLATS = 200000  # Increased for better coverage
TRAIN_BATCH = 8192  # Larger batches for stability

# ============================================================================
# DATA STRUCTURES
# ============================================================================

# G-Buffer
pos = ti.Vector.field(3, dtype=ti.f32, shape=(RES_X, RES_Y))
normal = ti.Vector.field(3, dtype=ti.f32, shape=(RES_X, RES_Y))
mat_id = ti.field(dtype=ti.i32, shape=(RES_X, RES_Y))
depth = ti.field(dtype=ti.f32, shape=(RES_X, RES_Y))

# ... existing buffers ...
# --- SHADOW PIPELINE BUFFERS ---
raw_shadow_map = ti.field(dtype=ti.f32, shape=(RES_X, RES_Y))  # Noisy input (0 or 1)
shadow_history = ti.field(dtype=ti.f32, shape=(RES_X, RES_Y))  # Temporal history
shadow_display = ti.field(dtype=ti.f32, shape=(RES_X, RES_Y))  # Final smooth shadow
shadow_temp = ti.field(dtype=ti.f32, shape=(RES_X, RES_Y))
# We don't need the PyTorch fields anymore!
# Data for Denoiser (Taichi -> PyTorch)
# The NN output
clean_accum_shadow = ti.field(
    dtype=ti.f32, shape=(RES_X, RES_Y)
)  # The "Ground Truth" accumulator
train_valid_mask = ti.field(
    dtype=ti.f32, shape=(RES_X, RES_Y)
)  # Where is the ground truth valid?

# ...

# Motion Vectors (screen-space)
motion_vector = ti.Vector.field(2, dtype=ti.f32, shape=(RES_X, RES_Y))
velocity_3d = ti.Vector.field(
    3, dtype=ti.f32, shape=(RES_X, RES_Y)
)  # World-space velocity

# Previous Frame Data
prev_pos = ti.Vector.field(3, dtype=ti.f32, shape=(RES_X, RES_Y))
prev_normal = ti.Vector.field(3, dtype=ti.f32, shape=(RES_X, RES_Y))
prev_mat_id = ti.field(dtype=ti.i32, shape=(RES_X, RES_Y))

# Lighting Buffers
direct_light = ti.Vector.field(3, dtype=ti.f32, shape=(RES_X, RES_Y))
indirect_light = ti.Vector.field(3, dtype=ti.f32, shape=(RES_X, RES_Y))
neural_out = ti.Vector.field(3, dtype=ti.f32, shape=(INTERNAL_X, INTERNAL_Y))

direct_accum_buffer = ti.Vector.field(3, dtype=ti.f32, shape=(RES_X, RES_Y))


# light denoising
filtered_direct = ti.Vector.field(3, dtype=ti.f32, shape=(RES_X, RES_Y))
indirect_light = ti.Vector.field(3, dtype=ti.f32, shape=(RES_X, RES_Y))
# Temporal Accumulation
accum_buffer = ti.Vector.field(3, dtype=ti.f32, shape=(RES_X, RES_Y))
accum_moments = ti.Vector.field(2, dtype=ti.f32, shape=(RES_X, RES_Y))  # mean, variance
confidence = ti.field(dtype=ti.f32, shape=(RES_X, RES_Y))
sample_count = ti.field(dtype=ti.i32, shape=(RES_X, RES_Y))

display_image = ti.Vector.field(3, dtype=ti.f32, shape=(RES_X, RES_Y))

# Gaussian Splat Cache for Indirect Lighting
splat_center = ti.Vector.field(3, dtype=ti.f32, shape=MAX_SPLATS)
splat_radiance = ti.Vector.field(3, dtype=ti.f32, shape=MAX_SPLATS)
splat_normal = ti.Vector.field(3, dtype=ti.f32, shape=MAX_SPLATS)
splat_radius = ti.field(dtype=ti.f32, shape=MAX_SPLATS)
splat_age = ti.field(dtype=ti.i32, shape=MAX_SPLATS)
splat_active = ti.field(dtype=ti.i32, shape=MAX_SPLATS)
splat_ptr = ti.field(dtype=ti.i32, shape=())

# ... inside Gaussian Splat Cache section ...
splat_active = ti.field(dtype=ti.i32, shape=MAX_SPLATS)
# ADD THIS LINE:
splat_obj_id = ti.field(dtype=ti.i32, shape=MAX_SPLATS)
splat_ptr = ti.field(dtype=ti.i32, shape=())

# Camera State
cam_pos = ti.Vector.field(3, dtype=ti.f32, shape=())
cam_lookat = ti.Vector.field(3, dtype=ti.f32, shape=())
cam_velocity = ti.Vector.field(3, dtype=ti.f32, shape=())

prev_cam_pos = ti.Vector.field(3, dtype=ti.f32, shape=())
prev_cam_lookat = ti.Vector.field(3, dtype=ti.f32, shape=())

# Scene Objects
light_pos = ti.Vector.field(3, dtype=ti.f32, shape=())
box_pos = ti.Vector.field(3, dtype=ti.f32, shape=())
box_vel = ti.Vector.field(3, dtype=ti.f32, shape=())
prev_box_pos = ti.Vector.field(3, dtype=ti.f32, shape=())

frame_idx = ti.field(dtype=ti.i32, shape=())

# ============================================================================
# ENHANCED NEURAL NETWORK
# ============================================================================


class ShadowDenoiser(nn.Module):
    """Tiny U-Net to clean noisy shadows using G-Buffer features"""

    def __init__(self):
        super().__init__()
        # Input: 1 (Shadow) + 3 (Normal) + 1 (Depth) + 2 (Screen Pos) = 7 channels
        self.enc1 = nn.Sequential(
            nn.Conv2d(7, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.pool2 = nn.MaxPool2d(2)

        self.center = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.ReLU()
        )

        self.up2 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.dec2 = nn.Sequential(
            nn.Conv2d(128 + 64, 64, kernel_size=3, padding=1), nn.ReLU()
        )

        self.up1 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.dec1 = nn.Sequential(
            nn.Conv2d(64 + 32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 1, kernel_size=1),
            nn.Sigmoid(),  # Shadow factor must be 0-1
        )

    def forward(self, shadow, normal, depth, coord):
        # x shape: [Batch, 7, H, W]
        x = torch.cat([shadow, normal, depth, coord], dim=1)

        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        c = self.center(self.pool2(e2))
        d2 = self.dec2(torch.cat([self.up2(c), e2], dim=1))
        out = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return out


class ImprovedNeuralGI(nn.Module):
    """Enhanced network with multi-scale features and attention"""

    def __init__(self, in_dim=36, hidden=64, out_dim=3):
        super().__init__()

        # Multi-scale feature extraction
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
        )

        # Attention-like mechanism for spatial awareness
        self.attention = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.SiLU(),
            nn.Linear(hidden // 2, hidden),
            nn.Sigmoid(),
        )

        # Output head
        self.decoder = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.LayerNorm(hidden // 2),
            nn.SiLU(),
            nn.Linear(hidden // 2, out_dim),
            nn.Softplus(),  # Ensure positive radiance
        )

        # Initialize output layer with small weights
        nn.init.uniform_(self.decoder[-2].weight, -0.001, 0.001)
        nn.init.constant_(self.decoder[-2].bias, 0.01)

    def forward(self, x):
        feat = self.encoder(x)
        attn = self.attention(feat)
        feat = feat * attn
        return self.decoder(feat)


# Network weights for Taichi kernels
W_enc1 = ti.field(dtype=ti.f32, shape=(64, 36))
b_enc1 = ti.field(dtype=ti.f32, shape=64)
W_enc2 = ti.field(dtype=ti.f32, shape=(64, 64))
b_enc2 = ti.field(dtype=ti.f32, shape=64)
W_out1 = ti.field(dtype=ti.f32, shape=(32, 64))
b_out1 = ti.field(dtype=ti.f32, shape=32)
W_out2 = ti.field(dtype=ti.f32, shape=(3, 32))
b_out2 = ti.field(dtype=ti.f32, shape=3)


# ============================================================================
# SCENE GEOMETRY
# ============================================================================
@ti.func
def sample_history_bilinear(u: float, v: float) -> float:
    """Reads history with CLAMPING to prevent black borders"""
    # CRITICAL FIX: Clamp u, v to valid range [0, Width-1.001]
    # This ensures x1 and y1 are never out of bounds!
    u = ti.max(0.0, ti.min(float(RES_X) - 1.001, u))
    v = ti.max(0.0, ti.min(float(RES_Y) - 1.001, v))

    x0 = int(ti.floor(u))
    y0 = int(ti.floor(v))
    x1 = x0 + 1
    y1 = y0 + 1

    wx = u - float(x0)
    wy = v - float(y0)

    # Safe to read now because we clamped above
    v00 = shadow_history[x0, y0]
    v10 = shadow_history[x1, y0]
    v01 = shadow_history[x0, y1]
    v11 = shadow_history[x1, y1]

    top = v00 * (1.0 - wx) + v10 * wx
    bot = v01 * (1.0 - wx) + v11 * wx

    return top * (1.0 - wy) + bot * wy


@ti.kernel
def init_scene():
    cam_pos[None] = [0, 0, 3.5]
    cam_lookat[None] = [0, 0, 0]
    cam_velocity[None] = [0, 0, 0]

    prev_cam_pos[None] = cam_pos[None]
    prev_cam_lookat[None] = cam_lookat[None]

    light_pos[None] = [0, 0.95, 0]
    box_pos[None] = [0, -0.5, 0]
    box_vel[None] = [1.2, 0.5, 0.4]
    prev_box_pos[None] = box_pos[None]

    frame_idx[None] = 0
    splat_ptr[None] = 0

    # Clear buffers
    for i, j in ti.ndrange(RES_X, RES_Y):
        sample_count[i, j] = 0
        confidence[i, j] = 0.0


@ti.kernel
def update_physics(dt: ti.f32):
    # Store previous
    prev_box_pos[None] = box_pos[None]

    # Update box
    box_pos[None] += box_vel[None] * dt

    # Random impulses
    if ti.random() < 0.02:
        box_vel[None] += (
            ti.Vector([ti.random() - 0.5, ti.random() - 0.5, ti.random() - 0.5]) * 2.0
        )

    # Boundaries with damping
    if box_pos[None].y < -0.6:
        box_pos[None].y = -0.6
        box_vel[None].y *= -0.8
    if box_pos[None].y > 0.6:
        box_pos[None].y = 0.6
        box_vel[None].y *= -0.8
    if ti.abs(box_pos[None].x) > 0.6:
        box_pos[None].x = 0.6 * (1 if box_pos[None].x > 0 else -1)
        box_vel[None].x *= -0.8
    if ti.abs(box_pos[None].z) > 0.6:
        box_pos[None].z = 0.6 * (1 if box_pos[None].z > 0 else -1)
        box_vel[None].z *= -0.8

    # Speed limit
    if box_vel[None].norm() > 3.0:
        box_vel[None] = box_vel[None].normalized() * 3.0


@ti.data_oriented
class SceneGeometry:
    def __init__(self):
        self.light_size = 0.15

    @ti.func
    def intersect(self, ro, rd):
        t_min = 1e9
        id_min = -1
        n_min = ti.Vector([0.0, 0.0, 0.0])

        # Walls (Cornell Box)
        walls = [
            ([-1.0, 0, 0], [1, 0, 0], 0),  # Left (red)
            ([1.0, 0, 0], [-1, 0, 0], 1),  # Right (green)
            ([0, -1.0, 0], [0, 1, 0], 2),  # Floor (white)
            ([0, 1.0, 0], [0, -1, 0], 3),  # Ceiling (white)
            ([0, 0, -1.0], [0, 0, 1], 4),  # Back (white)
        ]

        for i in ti.static(range(5)):
            plane_point = ti.Vector(walls[i][0])
            plane_normal = ti.Vector(walls[i][1])
            id = walls[i][2]

            denom = rd.dot(plane_normal)
            if ti.abs(denom) > 1e-6:
                t = (plane_point - ro).dot(plane_normal) / denom
                p = ro + rd * t

                # Check bounds
                valid = True
                if i < 2:  # X walls
                    valid = ti.abs(p.y) < 1.0 and ti.abs(p.z) < 1.0
                elif i < 4:  # Y walls
                    valid = ti.abs(p.x) < 1.0 and ti.abs(p.z) < 1.0
                else:  # Z wall
                    valid = ti.abs(p.x) < 1.0 and ti.abs(p.y) < 1.0

                if t > 0.001 and t < t_min and valid:
                    t_min = t
                    id_min = id
                    n_min = plane_normal

        # Animated Box (AABB)
        c = box_pos[None]
        half_size = 0.3
        b_min = c - half_size
        b_max = c + half_size

        t_near = -1e9
        t_far = 1e9

        for i in ti.static(range(3)):
            if ti.abs(rd[i]) < 1e-8:
                if ro[i] < b_min[i] or ro[i] > b_max[i]:
                    t_near = 1e9
            else:
                inv_d = 1.0 / rd[i]
                t1 = (b_min[i] - ro[i]) * inv_d
                t2 = (b_max[i] - ro[i]) * inv_d
                t_near = ti.max(t_near, ti.min(t1, t2))
                t_far = ti.min(t_far, ti.max(t1, t2))

        if t_far > t_near and t_far > 0.0:
            t = t_near if t_near > 0.001 else t_far
            if t > 0.001 and t < t_min:
                t_min = t
                id_min = 10
                hit_p = ro + rd * t

                # Determine face normal
                eps = 1e-3
                if ti.abs(hit_p.x - b_min.x) < eps:
                    n_min = [-1, 0, 0]
                elif ti.abs(hit_p.x - b_max.x) < eps:
                    n_min = [1, 0, 0]
                elif ti.abs(hit_p.y - b_min.y) < eps:
                    n_min = [0, -1, 0]
                elif ti.abs(hit_p.y - b_max.y) < eps:
                    n_min = [0, 1, 0]
                elif ti.abs(hit_p.z - b_min.z) < eps:
                    n_min = [0, 0, -1]
                else:
                    n_min = [0, 0, 1]

        return t_min, id_min, n_min

    @ti.func
    def get_albedo(self, id):
        c = ti.Vector([0.8, 0.8, 0.8])
        if id == 0:  # Left wall - red
            c = [0.8, 0.1, 0.1]
        elif id == 1:  # Right wall - green
            c = [0.1, 0.8, 0.1]
        elif id == 10:  # Box - blue
            c = [0.1, 0.5, 0.9]
        return c

    @ti.func
    def get_emission(self, id, p):
        emit = ti.Vector([0.0, 0.0, 0.0])
        # Ceiling light
        if id == 3:
            if (
                ti.abs(p.x - light_pos[None].x) < self.light_size
                and ti.abs(p.z - light_pos[None].z) < self.light_size
            ):
                emit = [20.0, 20.0, 20.0]
        return emit


scene = SceneGeometry()

# ============================================================================
# POSITIONAL ENCODING
# ============================================================================


@ti.func
def enhanced_encoding(p, n):
    """Enhanced positional encoding with normal information"""
    enc = ti.Vector([0.0] * 36)

    # Position encoding (6 frequencies)
    freqs = ti.Vector([1.0, 2.0, 4.0, 8.0, 16.0, 32.0])
    for i in ti.static(range(6)):
        base = i * 6
        enc[base + 0] = ti.sin(freqs[i] * p.x)
        enc[base + 1] = ti.cos(freqs[i] * p.x)
        enc[base + 2] = ti.sin(freqs[i] * p.y)
        enc[base + 3] = ti.cos(freqs[i] * p.y)
        enc[base + 4] = ti.sin(freqs[i] * p.z)
        enc[base + 5] = ti.cos(freqs[i] * p.z)

    return enc


# ============================================================================
# RENDERING KERNELS
# ============================================================================


@ti.func
def get_ray_direction(x, y, cam_p, cam_look):
    """Generate camera ray with proper aspect ratio"""
    u = (x / RES_X) * 2.0 - 1.0
    v = (y / RES_Y) * 2.0 - 1.0

    look = (cam_look - cam_p).normalized()
    right = ti.Vector([0, 1, 0]).cross(look).normalized()
    up = look.cross(right).normalized()

    aspect = RES_Y / RES_X
    fov_scale = 1.0

    ray_dir = (right * u * fov_scale + up * v * aspect * fov_scale + look).normalized()

    return ray_dir


@ti.kernel
def denoise_direct_lighting():
    """Spatial Bilateral Filter to smooth shadow grain without lag"""
    for x, y in filtered_direct:
        # Default to raw color
        result = direct_light[x, y]

        if mat_id[x, y] != -1:
            center_n = normal[x, y]
            center_p = pos[x, y]

            total_weight = 0.0
            total_color = ti.Vector([0.0, 0.0, 0.0])

            # 5x5 Blur Window
            for i in range(-2, 3):
                for j in range(-2, 3):
                    nx = x + i
                    ny = y + j

                    # Bounds check
                    if nx >= 0 and nx < RES_X and ny >= 0 and ny < RES_Y:
                        # 1. Spatial Weight (Gaussian) - Pixels closer are more important
                        dist_sq = float(i * i + j * j)
                        w_spatial = ti.exp(-dist_sq / 4.0)

                        # 2. Geometry Weight - Only blend if Normal/Depth are similar
                        # This preserves the sharp edges of the box!
                        n_diff = (center_n - normal[nx, ny]).norm()
                        p_diff = (center_p - pos[nx, ny]).norm()

                        if n_diff < 0.2 and p_diff < 0.1:
                            total_color += direct_light[nx, ny] * w_spatial
                            total_weight += w_spatial

            if total_weight > 0.0:
                result = total_color / total_weight

        filtered_direct[x, y] = result


@ti.kernel
def render_gbuffer():
    """Render G-Buffer and compute motion vectors"""
    for x, y in pos:
        # Current frame ray
        ro = cam_pos[None]
        rd = get_ray_direction(x, y, cam_pos[None], cam_lookat[None])

        t, id, n = scene.intersect(ro, rd)

        if id != -1:
            hit_pos = ro + rd * t
            pos[x, y] = hit_pos
            normal[x, y] = n
            mat_id[x, y] = id
            depth[x, y] = t

            # Compute 3D velocity for this point
            vel_3d = ti.Vector([0.0, 0.0, 0.0])
            if id == 10:  # Moving box
                vel_3d = box_vel[None]

            velocity_3d[x, y] = vel_3d

            # Compute motion vector (screen-space)
            # Previous position
            prev_world_pos = hit_pos - vel_3d * 0.016  # Assume 60fps

            # Project to previous screen space
            prev_ro = prev_cam_pos[None]
            prev_look = (prev_cam_lookat[None] - prev_ro).normalized()
            prev_right = ti.Vector([0, 1, 0]).cross(prev_look).normalized()
            prev_up = prev_look.cross(prev_right).normalized()

            view_vec = prev_world_pos - prev_ro
            dist = view_vec.dot(prev_look)

            if dist > 0.1:
                proj_right = view_vec.dot(prev_right) / dist
                proj_up = view_vec.dot(prev_up) / dist
                aspect = RES_Y / RES_X

                prev_u = proj_right
                prev_v = proj_up / aspect

                prev_x = (prev_u + 1.0) * 0.5 * RES_X
                prev_y = (prev_v + 1.0) * 0.5 * RES_Y

                motion_vector[x, y] = ti.Vector([prev_x - x, prev_y - y])
            else:
                motion_vector[x, y] = ti.Vector([0.0, 0.0])
        else:
            mat_id[x, y] = -1
            depth[x, y] = 1e9
            motion_vector[x, y] = ti.Vector([0.0, 0.0])


@ti.func
def fract(x):
    """Helper to get fractional part of a number (like GLSL fract)"""
    return x - ti.floor(x)


@ti.func
def ign_noise(x, y, frame):
    """Interleaved Gradient Noise: High frequency, easy to denoise"""
    # Fixed: Replaced ti.fract with local helper
    return fract(
        52.9829189
        * fract(0.06711056 * float(x) + 0.00583715 * float(y) + 0.006237 * float(frame))
    )


@ti.kernel
def render_direct_lighting():
    """Trace 4 Rotated Rays with Higher Bias"""
    for x, y in raw_shadow_map:
        if mat_id[x, y] == -1:
            raw_shadow_map[x, y] = 1.0
        else:
            p = pos[x, y]
            n = normal[x, y]

            # Rotation
            theta = ti.random() * 6.283185
            cos_t = ti.cos(theta)
            sin_t = ti.sin(theta)

            total_vis = 0.0

            for i in range(2):
                for j in range(2):
                    u_base = (float(i) + 0.5) / 2.0 - 0.5
                    v_base = (float(j) + 0.5) / 2.0 - 0.5
                    u = u_base + (ti.random() - 0.5) * 0.5
                    v = v_base + (ti.random() - 0.5) * 0.5

                    u_rot = u * cos_t - v * sin_t
                    v_rot = u * sin_t + v * cos_t

                    off_x = u_rot * scene.light_size * 2.0
                    off_z = v_rot * scene.light_size * 2.0

                    l_sample = light_pos[None] + ti.Vector([off_x, 0.0, off_z])
                    l_vec = l_sample - p
                    dist = l_vec.norm()
                    dir = l_vec.normalized()

                    # INCREASED BIAS: 1e-3 -> 5e-3
                    t, _, _ = scene.intersect(p + n * 5e-3, dir)

                    if t < 0 or t > dist - 0.01:
                        total_vis += 1.0

            raw_shadow_map[x, y] = total_vis / 4.0


@ti.kernel
def filter_shadow_prepass():
    """Pass 0: 5x5 Pre-Blur to create a stable target for rejection"""
    for x, y in shadow_temp:
        if mat_id[x, y] == -1:
            shadow_temp[x, y] = 1.0
        else:
            center_n = normal[x, y]
            center_d = depth[x, y]
            sum_val = 0.0
            sum_w = 0.0

            # Widen to 5x5 (-2 to 2)
            for i in range(-2, 3):
                for j in range(-2, 3):
                    nx = x + i
                    ny = y + j
                    if nx >= 0 and nx < RES_X and ny >= 0 and ny < RES_Y:
                        # Relaxed depth check to smooth out noise
                        n_diff = (normal[nx, ny] - center_n).norm()
                        d_diff = ti.abs(depth[nx, ny] - center_d)

                        if n_diff < 0.2 and d_diff < 0.1:
                            sum_val += raw_shadow_map[nx, ny]
                            sum_w += 1.0

            shadow_temp[x, y] = sum_val / ti.max(1.0, sum_w)


@ti.kernel
def filter_shadow_temporal():
    """Pass 1: AABB Variance Clipping (The Stable Standard)"""
    for x, y in shadow_history:
        if mat_id[x, y] == -1:
            shadow_history[x, y] = 1.0
        else:
            # 1. Neighborhood Statistics (5x5)
            m1 = 0.0
            m2 = 0.0
            count = 0.0
            for i in range(-2, 3):
                for j in range(-2, 3):
                    nx = ti.max(0, ti.min(x + i, RES_X - 1))
                    ny = ti.max(0, ti.min(y + j, RES_Y - 1))
                    val = shadow_temp[nx, ny]  # Read PRE-PASS buffer
                    m1 += val
                    m2 += val * val
                    count += 1.0

            mu = m1 / count
            sigma = ti.sqrt(ti.max(0.0, m2 / count - mu * mu))

            # AABB Clamp Box
            # 1.2 sigma is tighter than 1.5, reduces ghosting
            min_c = mu - 1.2 * sigma
            max_c = mu + 1.2 * sigma

            curr_vis = shadow_temp[x, y]

            # 2. Reproject
            motion = motion_vector[x, y]
            prev_u = float(x) + motion.x
            prev_v = float(y) + motion.y

            valid = 0
            hist_vis = curr_vis

            # Simplified Validity Check (Bounds checked inside sampler now)
            if (
                prev_u >= -1.0
                and prev_u < float(RES_X) + 1.0
                and prev_v >= -1.0
                and prev_v < float(RES_Y) + 1.0
            ):
                # Check ID at the rounded coordinate
                prev_ix = int(prev_u)
                prev_iy = int(prev_v)
                # Safety clamp for ID check
                prev_ix = ti.max(0, ti.min(RES_X - 1, prev_ix))
                prev_iy = ti.max(0, ti.min(RES_Y - 1, prev_iy))

                if prev_mat_id[prev_ix, prev_iy] == mat_id[x, y]:
                    valid = 1
                    hist_vis = sample_history_bilinear(prev_u, prev_v)

            # 3. Clip History
            if valid:
                hist_vis = ti.max(min_c, ti.min(max_c, hist_vis))

            # 4. Blend
            # 90% History (Standard)
            alpha = 0.1
            if valid == 0:
                alpha = 1.0

            shadow_history[x, y] = hist_vis * (1.0 - alpha) + curr_vis * alpha


@ti.kernel
def gather_training_samples():
    """Gather indirect lighting training samples via path tracing"""
    for sample_idx in range(TRAIN_BATCH // 4):
        # Random pixel
        x = int(ti.random() * RES_X)
        y = int(ti.random() * RES_Y)

        # Check if valid surface
        valid = 0
        if x >= 0 and x < RES_X and y >= 0 and y < RES_Y:
            if mat_id[x, y] != -1:
                valid = 1

        if valid == 1:
            p = pos[x, y]
            n = normal[x, y]

            # Cosine-weighted hemisphere sample
            r1 = ti.random()
            r2 = ti.random()
            phi = 2.0 * math.pi * r1
            cos_theta = ti.sqrt(1.0 - r2)
            sin_theta = ti.sqrt(r2)

            # Local coordinate system
            up = ti.Vector([0, 1, 0])
            if ti.abs(n.y) >= 0.9:
                up = ti.Vector([1, 0, 0])

            tangent = up.cross(n).normalized()
            bitangent = n.cross(tangent)

            local_dir = ti.Vector(
                [sin_theta * ti.cos(phi), cos_theta, sin_theta * ti.sin(phi)]
            )

            world_dir = (
                tangent * local_dir.x + n * local_dir.y + bitangent * local_dir.z
            ).normalized()

            # Trace bounce ray
            t_bounce, id_bounce, n_bounce = scene.intersect(p + n * 1e-3, world_dir)

            radiance = ti.Vector([0.0, 0.0, 0.0])

            if id_bounce != -1:
                p_bounce = p + world_dir * t_bounce
                albedo_bounce = scene.get_albedo(id_bounce)
                emit_bounce = scene.get_emission(id_bounce, p_bounce)

                if emit_bounce.norm() > 0.1:
                    # Hit light directly
                    radiance = emit_bounce
                else:
                    # Sample light from bounce point
                    light_vec = light_pos[None] - p_bounce
                    light_dist = light_vec.norm()
                    light_dir = light_vec.normalized()

                    t_shadow, _, _ = scene.intersect(
                        p_bounce + n_bounce * 1e-3, light_dir
                    )

                    vis = 0.0
                    if t_shadow < 0 or t_shadow > light_dist - 0.01:
                        vis = 1.0

                    cos_theta_light = ti.max(0.0, n_bounce.dot(light_dir))
                    light_intensity = 50.0
                    falloff = 1.0 / (light_dist * light_dist + 0.1)
                    radiance = (
                        albedo_bounce
                        * vis
                        * cos_theta_light
                        * light_intensity
                        * falloff
                    )

            # Store in splat cache if valid
            if radiance.norm() > 0.01:
                idx = ti.atomic_add(splat_ptr[None], 1) % MAX_SPLATS
                splat_center[idx] = p
                splat_radiance[idx] = ti.log(1.0 + radiance) / 4.0  # Compressed
                splat_normal[idx] = n
                splat_radius[idx] = 0.1
                splat_age[idx] = frame_idx[None]
                splat_active[idx] = 1
                splat_obj_id[idx] = mat_id[x, y]


@ti.kernel
def render_neural_indirect():
    """Use neural network to predict indirect lighting"""
    for x, y in neural_out:
        result = ti.Vector([0.0, 0.0, 0.0])

        # Map to full resolution G-buffer
        fx = int(x * (RES_X / INTERNAL_X))
        fy = int(y * (RES_Y / INTERNAL_Y))

        if mat_id[fx, fy] != -1:
            p = pos[fx, fy]
            n = normal[fx, fy]

            # Enhanced encoding
            enc = enhanced_encoding(p, n)

            # Manual forward pass (Layer 1) - 64 units
            h1 = ti.Vector([0.0] * 64)
            for i in range(64):
                val = b_enc1[i]
                for j in range(36):
                    val += W_enc1[i, j] * enc[j]
                # SiLU activation
                h1[i] = val / (1.0 + ti.exp(-val))

            # Layer 2 - 64 units
            h2 = ti.Vector([0.0] * 64)
            for i in range(64):
                val = b_enc2[i]
                for j in range(64):
                    val += W_enc2[i, j] * h1[j]
                h2[i] = val / (1.0 + ti.exp(-val))

            # Output layer 1 - 32 units
            h3 = ti.Vector([0.0] * 32)
            for i in range(32):
                val = b_out1[i]
                for j in range(64):
                    val += W_out1[i, j] * h2[j]
                h3[i] = val / (1.0 + ti.exp(-val))

            # Output layer 2 - 3 units
            out = ti.Vector([0.0, 0.0, 0.0])
            for i in range(3):
                val = b_out2[i]
                for j in range(32):
                    val += W_out2[i, j] * h3[j]
                out[i] = ti.log(1.0 + ti.max(0.0, val))  # Softplus

            # Decompress
            indirect = ti.exp(out * 4.0) - 1.0
            albedo = scene.get_albedo(mat_id[fx, fy])

            result = albedo * indirect

        neural_out[x, y] = result


@ti.kernel
def temporal_accumulation_with_prediction():
    for x, y in display_image:
        if mat_id[x, y] != -1:
            # 1. UPSAMPLE INDIRECT (Unchanged)
            ux = x * (INTERNAL_X / RES_X)
            uy = y * (INTERNAL_Y / RES_Y)
            x0 = int(ux)
            y0 = int(uy)
            x1 = ti.min(x0 + 1, INTERNAL_X - 1)
            y1 = ti.min(y0 + 1, INTERNAL_Y - 1)
            wx = ux - x0
            wy = uy - y0
            indirect_upsampled = (
                neural_out[x0, y0] * (1 - wx) * (1 - wy)
                + neural_out[x1, y0] * wx * (1 - wy)
                + neural_out[x0, y1] * (1 - wx) * wy
                + neural_out[x1, y1] * wx * wy
            )

            # 2. REPROJECTION
            motion = motion_vector[x, y]
            prev_x = int(x + motion.x)
            prev_y = int(y + motion.y)

            valid = 0
            hist_indirect = ti.Vector([0.0, 0.0, 0.0])
            hist_direct = ti.Vector([0.0, 0.0, 0.0])  # New history var
            hist_conf = 0.0

            if prev_x >= 0 and prev_x < RES_X and prev_y >= 0 and prev_y < RES_Y:
                # Strict depth/normal check
                if prev_mat_id[prev_x, prev_y] == mat_id[x, y]:
                    depth_diff = ti.abs(depth[x, y] - depth[prev_x, prev_y])
                    n_diff = (normal[x, y] - prev_normal[prev_x, prev_y]).norm()

                    if depth_diff < 0.1 and n_diff < 0.2:
                        valid = 1
                        hist_indirect = accum_buffer[prev_x, prev_y]
                        hist_direct = direct_accum_buffer[
                            prev_x, prev_y
                        ]  # Fetch shadow history
                        hist_conf = confidence[prev_x, prev_y]

            # 3. BLEND FACTORS
            alpha_indirect = 0.1  # Default 90% history
            alpha_direct = 0.2  # Default 80% history (stops flicker)

            # CRITICAL: GHOSTING FIX
            # If pixel is moving fast, reject shadow history to stop trails
            motion_mag = motion.norm()
            if motion_mag > 2.0:
                alpha_direct = 0.8  # Drop to 20% history (noisy but no trails)
                alpha_indirect = 0.5

            if valid == 0:
                alpha_indirect = 1.0
                alpha_direct = 1.0

            # 4. BLEND
            new_indirect = (
                hist_indirect * (1.0 - alpha_indirect)
                + indirect_upsampled * alpha_indirect
            )

            # This blends the shadows to kill the flicker!
            new_direct = (
                hist_direct * (1.0 - alpha_direct) + direct_light[x, y] * alpha_direct
            )

            # Store histories
            accum_buffer[x, y] = new_indirect
            direct_accum_buffer[x, y] = new_direct  # Store clean shadow

            confidence[x, y] = ti.min(1.0, hist_conf + 0.05) if valid else 0.1

            # 5. COMPOSITE
            # Use the SMOOTHED direct light + SMOOTHED indirect
            final_color = new_direct + new_indirect

            tone_mapped = final_color / (final_color + 1.0)
            display_image[x, y] = ti.pow(tone_mapped, 1.0 / 2.2)


@ti.kernel
def copy_prev_frame():
    """Store current frame data for next frame's reprojection"""
    for i, j in pos:
        prev_pos[i, j] = pos[i, j]
        prev_normal[i, j] = normal[i, j]
        prev_mat_id[i, j] = mat_id[i, j]


@ti.kernel
def age_splats():
    """Age out old splats"""
    for i in range(MAX_SPLATS):
        if splat_active[i] == 1:
            age = frame_idx[None] - splat_age[i]
            if age > 180:  # 3 seconds at 60fps
                splat_active[i] = 0


# ============================================================================
# TRAINING
# ============================================================================
@ti.kernel
def accumulate_ground_truth_kernel():
    # Simple exponential moving average for ground truth
    # Only runs when static, so no motion vectors needed!
    for x, y in clean_accum_shadow:
        clean_accum_shadow[x, y] = (
            clean_accum_shadow[x, y] * 0.95 + raw_shadow_map[x, y] * 0.05
        )


@ti.kernel
def reset_accumulator_kernel():
    for x, y in clean_accum_shadow:
        clean_accum_shadow[x, y] = raw_shadow_map[x, y]  # Reset to current


@ti.kernel
def composite_kernel():
    for x, y in display_image:
        if mat_id[x, y] != -1:
            # Reconstruct Direct Light
            # We have the visibility from the Neural Net (0.0 to 1.0)
            vis = denoised_shadow_map[x, y]

            # Recalculate lighting intensity (without the shadow ray)
            # (You can store this 'unshadowed_light' in a buffer earlier to save perf)
            p = pos[x, y]
            n = normal[x, y]
            light_vec = light_pos[None] - p
            dist = light_vec.norm()
            intensity = (
                50.0
                * (1.0 / (dist * dist + 0.1))
                * ti.max(0.0, n.dot(light_vec.normalized()))
            )

            direct = scene.get_albedo(mat_id[x, y]) * intensity * vis

            # Add Neural Indirect
            final = (
                direct + accum_buffer[x, y]
            )  # The indirect buffer from previous steps

            # Tone map
            display_image[x, y] = ti.pow(final / (final + 1.0), 1.0 / 2.2)


@ti.kernel
def resolve_frame():
    for x, y in display_image:
        if mat_id[x, y] == -1:
            display_image[x, y] = ti.Vector([0.05, 0.05, 0.05])
        else:
            # USE THE CLEAN SHADOW BUFFER
            vis = shadow_display[x, y]

            # Direct Light Calculation
            p = pos[x, y]
            n = normal[x, y]
            albedo = scene.get_albedo(mat_id[x, y])
            light_vec = light_pos[None] - p
            dist_sq = light_vec.norm_sqr()
            light_dir = light_vec.normalized()
            intensity = 50.0 * (1.0 / (dist_sq + 0.1)) * ti.max(0.0, n.dot(light_dir))

            direct_color = albedo * intensity * vis

            # Indirect Light (Temporal Accumulation)
            # ... (Keep your existing indirect accumulation logic here) ...
            # (Copy from previous answers, just ensuring 'direct_color' uses 'vis' from above)

            # Recopying simplified indirect logic for completeness:
            ux = x * (INTERNAL_X / RES_X)
            uy = y * (INTERNAL_Y / RES_Y)
            x0 = int(ux)
            y0 = int(uy)
            indirect_upsampled = neural_out[x0, y0]

            motion = motion_vector[x, y]
            prev_x = int(x + motion.x)
            prev_y = int(y + motion.y)
            hist_indirect = ti.Vector([0.0, 0.0, 0.0])
            valid = 0
            if prev_x >= 0 and prev_x < RES_X and prev_y >= 0 and prev_y < RES_Y:
                if prev_mat_id[prev_x, prev_y] == mat_id[x, y]:
                    valid = 1
                    hist_indirect = accum_buffer[prev_x, prev_y]

            alpha = 0.1 if valid else 1.0
            final_indirect = hist_indirect * (1.0 - alpha) + indirect_upsampled * alpha
            accum_buffer[x, y] = final_indirect

            # Composite
            final = direct_color + final_indirect
            tone_mapped = final / (final + 1.0)
            display_image[x, y] = ti.pow(tone_mapped, 1.0 / 2.2)


@ti.kernel
def advect_splats(dt: ti.f32):
    """Move splats that are attached to the moving box"""
    for i in range(MAX_SPLATS):
        # Check if splat is active AND belongs to the box (ID 10)
        # Note: You must ensure splat_obj_id is defined in Data Structures
        if splat_active[i] == 1 and splat_obj_id[i] == 10:
            splat_center[i] += box_vel[None] * dt


@ti.kernel
def filter_shadow_spatial():
    """Pass 2: 7x7 Strong Spatial Blur"""
    for x, y in shadow_display:
        if mat_id[x, y] == -1:
            shadow_display[x, y] = 1.0
        else:
            center_n = normal[x, y]
            center_d = depth[x, y]

            sum_val = 0.0
            sum_weight = 0.0

            # 7x7 Blur
            for i in range(-3, 4):
                for j in range(-3, 4):
                    nx = x + i
                    ny = y + j
                    if nx >= 0 and nx < RES_X and ny >= 0 and ny < RES_Y:
                        # 1. Weights
                        n_diff = (normal[nx, ny] - center_n).norm()
                        d_diff = ti.abs(depth[nx, ny] - center_d)

                        # Strong edge stopping (keep box sharp)
                        w_geom = ti.exp(-n_diff * 4.0 - d_diff * 4.0)

                        # Flat Gaussian (Blur everything else)
                        w_dist = ti.exp(-(i * i + j * j) / 8.0)

                        weight = w_geom * w_dist

                        sum_val += shadow_history[nx, ny] * weight
                        sum_weight += weight

            shadow_display[x, y] = sum_val / (sum_weight + 1e-5)


@ti.kernel
def fill_training_batch(inp: ti.types.ndarray(), tgt: ti.types.ndarray()):
    """
    STABILIZED Self-Supervision:
    Prevents color explosions by blending History (Clean) with Splats (True).
    """
    for i in range(TRAIN_BATCH):
        idx = int(ti.random() * MAX_SPLATS)

        if splat_active[idx] == 1:
            p = splat_center[idx]
            n = splat_normal[idx]

            # 1. Input Encoding
            enc = enhanced_encoding(p, n)
            for k in range(36):
                inp[i, k] = enc[k]

            # 2. Project to find History
            cam_p = cam_pos[None]
            cam_l = cam_lookat[None]
            view_vec = p - cam_p
            look = (cam_l - cam_p).normalized()
            dist = view_vec.dot(look)

            # Default: Trust the RAW splat (Ground Truth)
            # We start with the noisy reality.
            raw_target = splat_radiance[idx]
            training_target = raw_target

            if dist > 0.1:
                right = ti.Vector([0, 1, 0]).cross(look).normalized()
                up = look.cross(right).normalized()
                aspect = RES_Y / RES_X
                u = view_vec.dot(right) / dist
                v = (view_vec.dot(up) / dist) / aspect

                sx = int((u + 1.0) * 0.5 * RES_X)
                sy = int((v + 1.0) * 0.5 * RES_Y)

                if sx >= 0 and sx < RES_X and sy >= 0 and sy < RES_Y:
                    d_buf = depth[sx, sy]
                    if ti.abs(dist - d_buf) < 0.2:
                        # Fetch History
                        hist_color = accum_buffer[sx, sy]

                        # SAFETY 1: Clamp History
                        # Prevent the "Green Explosion" by killing invalid values immediately
                        hist_color = ti.max(0.0, ti.min(10.0, hist_color))

                        # SAFETY 2: Log Space
                        # Networks learn better in Log domain (dampens huge lights)
                        hist_log = ti.log(1.0 + hist_color)
                        raw_log = ti.log(1.0 + raw_target)

                        # SAFETY 3: The Blend (Crucial!)
                        # Don't trust history 100%. If history is crazy green but
                        # raw splat is red, pull it back towards reality.
                        # 80% Clean History + 20% Noisy Truth
                        training_target = hist_log * 0.8 + raw_log * 0.2

            # Final Sanity Clamp before writing to PyTorch
            training_target = ti.max(0.0, ti.min(5.0, training_target))

            tgt[i, 0] = training_target.x
            tgt[i, 1] = training_target.y
            tgt[i, 2] = training_target.z


def sync_network_weights(model):
    """Copy PyTorch weights to Taichi fields"""
    W_enc1.from_numpy(model.encoder[0].weight.detach().cpu().numpy())
    b_enc1.from_numpy(model.encoder[0].bias.detach().cpu().numpy())
    W_enc2.from_numpy(model.encoder[3].weight.detach().cpu().numpy())
    b_enc2.from_numpy(model.encoder[3].bias.detach().cpu().numpy())
    W_out1.from_numpy(model.decoder[0].weight.detach().cpu().numpy())
    b_out1.from_numpy(model.decoder[0].bias.detach().cpu().numpy())
    W_out2.from_numpy(model.decoder[3].weight.detach().cpu().numpy())
    b_out2.from_numpy(model.decoder[3].bias.detach().cpu().numpy())


# ============================================================================
# MAIN LOOP
# ============================================================================


def main():
    print("Initializing Hybrid Neural Lumen...")

    # 1. SETUP NEURAL NET (Only for GI now)
    model = ImprovedNeuralGI(in_dim=36, hidden=64, out_dim=3).cuda()
    optimizer = optim.AdamW(model.parameters(), lr=0.003, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=1000, eta_min=0.0001
    )
    train_input = torch.zeros((TRAIN_BATCH, 36), device="cuda")
    train_target = torch.zeros((TRAIN_BATCH, 3), device="cuda")

    gui = ti.GUI("Neural Lumen - Hybrid Stable", res=(RES_X, RES_Y))
    init_scene()

    frame = 0
    paused = False
    training_enabled = True

    while gui.running:
        for e in gui.get_events(ti.GUI.PRESS):
            if e.key == ti.GUI.SPACE:
                paused = not paused
            elif e.key == "t":
                training_enabled = not training_enabled

        # CONTROLS
        prev_cam_pos[None] = cam_pos[None]
        prev_cam_lookat[None] = cam_lookat[None]
        speed = 0.05
        if gui.is_pressed("w"):
            cam_pos[None].z -= speed
            cam_lookat[None].z -= speed
        if gui.is_pressed("s"):
            cam_pos[None].z += speed
            cam_lookat[None].z += speed
        if gui.is_pressed("a"):
            cam_pos[None].x -= speed
            cam_lookat[None].x -= speed
        if gui.is_pressed("d"):
            cam_pos[None].x += speed
            cam_lookat[None].x += speed
        if gui.is_pressed("q"):
            cam_pos[None].y -= speed
            cam_lookat[None].y -= speed
        if gui.is_pressed("e"):
            cam_pos[None].y += speed
            cam_lookat[None].y += speed
        cam_velocity[None] = (cam_pos[None] - prev_cam_pos[None]) / 0.016

        if not paused:
            update_physics(0.016)
            advect_splats(0.016)

        frame_idx[None] = frame

        # --- RENDERING PIPELINE ---

        # 1. Geometry
        render_gbuffer()

        # 2. Shadows (The New Hybrid Fix)
        render_direct_lighting()  # 4 Rays (Stable Noise)
        filter_shadow_prepass()  # Pre-Blur (Calm Input)
        filter_shadow_temporal()  # Accumulate (Converge Static)
        filter_shadow_spatial()  # Clean noise

        # 3. Indirect Light (Neural)
        if frame % 2 == 0:
            gather_training_samples()
            age_splats()

        if training_enabled and frame > 5 and frame % 1 == 0:
            fill_training_batch(train_input, train_target)
            optimizer.zero_grad()
            pred = model(train_input)
            loss = nn.MSELoss()(pred, train_target)
            loss.backward()
            optimizer.step()
            scheduler.step()
            sync_network_weights(model)

        render_neural_indirect()

        # 4. Composite
        resolve_frame()
        copy_prev_frame()

        gui.set_image(display_image)
        gui.show()
        frame += 1

    print("Shutting down...")


if __name__ == "__main__":
    main()
