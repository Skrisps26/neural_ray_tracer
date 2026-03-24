import math
import time

import numpy as np
import taichi as ti
import torch
import torch.nn as nn
import torch.optim as optim

# --- 1. CONFIGURATION ---
ti.init(arch=ti.cuda, device_memory_fraction=0.9)
RES_X, RES_Y = 1280, 720
INTERNAL_X, INTERNAL_Y = 640, 360
MAX_SPLATS = 100000
TRAIN_BATCH = 4096

# --- 2. DATA STRUCTURES ---
pos = ti.Vector.field(3, dtype=ti.f32, shape=(RES_X, RES_Y))
normal = ti.Vector.field(3, dtype=ti.f32, shape=(RES_X, RES_Y))
mat_id = ti.field(dtype=ti.i32, shape=(RES_X, RES_Y))

neural_out = ti.Vector.field(3, dtype=ti.f32, shape=(INTERNAL_X, INTERNAL_Y))
prev_image = ti.Vector.field(3, dtype=ti.f32, shape=(RES_X, RES_Y))
display_image = ti.Vector.field(3, dtype=ti.f32, shape=(RES_X, RES_Y))

# The Cache
s_pos = ti.Vector.field(3, dtype=ti.f32, shape=MAX_SPLATS)
s_rad = ti.Vector.field(3, dtype=ti.f32, shape=MAX_SPLATS)
s_active = ti.field(dtype=ti.i32, shape=MAX_SPLATS)
splat_ptr = ti.field(dtype=ti.i32, shape=())

# Scene State
cam_pos = ti.Vector.field(3, dtype=ti.f32, shape=())
cam_lookat = ti.Vector.field(3, dtype=ti.f32, shape=())
prev_cam_pos = ti.Vector.field(3, dtype=ti.f32, shape=())  # FOR REPROJECTION
prev_cam_lookat = ti.Vector.field(3, dtype=ti.f32, shape=())  # FOR REPROJECTION

light_pos = ti.Vector.field(3, dtype=ti.f32, shape=())
box_pos = ti.Vector.field(3, dtype=ti.f32, shape=())
box_vel = ti.Vector.field(3, dtype=ti.f32, shape=())
frame_idx = ti.field(dtype=ti.i32, shape=())


# --- 3. NEURAL NETWORK ---
class NeuralCore(nn.Module):
    def __init__(self, out_dim=3, hidden=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(24, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_dim),
            nn.ReLU(),
        )
        nn.init.uniform_(self.net[-2].weight, -0.01, 0.01)
        nn.init.constant_(self.net[-2].bias, 0.01)

    def forward(self, x):
        return self.net(x)


W1 = ti.field(dtype=ti.f32, shape=(32, 24))
b1 = ti.field(dtype=ti.f32, shape=(32))
W2 = ti.field(dtype=ti.f32, shape=(32, 32))
b2 = ti.field(dtype=ti.f32, shape=(32))
W3 = ti.field(dtype=ti.f32, shape=(3, 32))
b3 = ti.field(dtype=ti.f32, shape=(3))


# --- 4. PHYSICS & GEOMETRY ---
@ti.kernel
def init_scene():
    cam_pos[None] = [0, 0, 3.5]
    cam_lookat[None] = [0, 0, 0]
    prev_cam_pos[None] = [0, 0, 3.5]
    prev_cam_lookat[None] = [0, 0, 0]

    light_pos[None] = [0, 0.9, 0]
    box_pos[None] = [0, -0.5, 0]
    box_vel[None] = [1.2, 0.5, 0.4]


@ti.kernel
def update_physics(dt: ti.f32):
    box_pos[None] += box_vel[None] * dt
    if ti.random() < 0.02:
        box_vel[None] += (
            ti.Vector([ti.random() - 0.5, ti.random() - 0.5, ti.random() - 0.5]) * 2.0
        )

    # Bounces
    if box_pos[None].y < -0.6:
        box_pos[None].y = -0.6
        box_vel[None].y *= -1.0
    if box_pos[None].y > 0.6:
        box_pos[None].y = 0.6
        box_vel[None].y *= -1.0
    if abs(box_pos[None].x) > 0.6:
        box_pos[None].x = 0.6 * (1 if box_pos[None].x > 0 else -1)
        box_vel[None].x *= -1.0
    if abs(box_pos[None].z) > 0.6:
        box_pos[None].z = 0.6 * (1 if box_pos[None].z > 0 else -1)
        box_vel[None].z *= -1.0
    if box_vel[None].norm() > 3.0:
        box_vel[None] = box_vel[None].normalized() * 3.0


@ti.data_oriented
class SceneGeometry:
    def __init__(self):
        self.light_size = 0.4

    @ti.func
    def intersect(self, ro, rd):
        t_min = 1e9
        id_min = -1
        n_min = ti.Vector([0.0, 0.0, 0.0])

        # Walls
        t = (-1.0 - ro.x) / rd.x
        p = ro + rd * t
        if t > 0.001 and t < t_min and abs(p.y) < 1 and abs(p.z) < 1:
            t_min = t
            id_min = 0
            n_min = [1, 0, 0]
        t = (1.0 - ro.x) / rd.x
        p = ro + rd * t
        if t > 0.001 and t < t_min and abs(p.y) < 1 and abs(p.z) < 1:
            t_min = t
            id_min = 1
            n_min = [-1, 0, 0]
        t = (-1.0 - ro.y) / rd.y
        p = ro + rd * t
        if t > 0.001 and t < t_min and abs(p.x) < 1 and abs(p.z) < 1:
            t_min = t
            id_min = 2
            n_min = [0, 1, 0]
        t = (1.0 - ro.y) / rd.y
        p = ro + rd * t
        if t > 0.001 and t < t_min and abs(p.x) < 1 and abs(p.z) < 1:
            t_min = t
            id_min = 3
            n_min = [0, -1, 0]
        t = (-1.0 - ro.z) / rd.z
        p = ro + rd * t
        if t > 0.001 and t < t_min and abs(p.x) < 1 and abs(p.y) < 1:
            t_min = t
            id_min = 4
            n_min = [0, 0, 1]

        # Box
        c = box_pos[None]
        b_min = c - 0.4
        b_max = c + 0.4
        t_box_near = -1e9
        t_box_far = 1e9
        for i in ti.static(range(3)):
            if abs(rd[i]) < 1e-6:
                if ro[i] < b_min[i] or ro[i] > b_max[i]:
                    t_box_near = 1e9
            else:
                inv_d = 1.0 / rd[i]
                t1 = (b_min[i] - ro[i]) * inv_d
                t2 = (b_max[i] - ro[i]) * inv_d
                t_box_near = max(t_box_near, min(t1, t2))
                t_box_far = min(t_box_far, max(t1, t2))

        if t_box_far > t_box_near and t_box_far > 0.0:
            t = t_box_near if t_box_near > 0.001 else t_box_far
            if t > 0.001 and t < t_min:
                t_min = t
                id_min = 10
                hit_p = ro + rd * t
                eps = 1e-3
                if abs(hit_p.x - b_min.x) < eps:
                    n_min = [-1, 0, 0]
                elif abs(hit_p.x - b_max.x) < eps:
                    n_min = [1, 0, 0]
                elif abs(hit_p.y - b_min.y) < eps:
                    n_min = [0, -1, 0]
                elif abs(hit_p.y - b_max.y) < eps:
                    n_min = [0, 1, 0]
                elif abs(hit_p.z - b_min.z) < eps:
                    n_min = [0, 0, -1]
                elif abs(hit_p.z - b_max.z) < eps:
                    n_min = [0, 0, 1]

        return t_min, id_min, n_min

    @ti.func
    def get_color(self, id):
        c = ti.Vector([0.0, 0.0, 0.0])
        if id == 0:
            c = [0.8, 0.1, 0.1]
        elif id == 1:
            c = [0.1, 0.8, 0.1]
        elif id == 10:
            c = [0.1, 0.4, 0.9]
        else:
            c = [0.9, 0.9, 0.9]
        return c


scene = SceneGeometry()


# --- 5. RENDER KERNELS ---
@ti.func
def positional_encoding(p):
    enc = ti.Vector([0.0] * 24)
    freqs = ti.Vector([1.0, 2.0, 4.0, 8.0])
    for i in ti.static(range(4)):
        enc[i * 6 + 0] = ti.sin(freqs[i] * p.x)
        enc[i * 6 + 1] = ti.cos(freqs[i] * p.x)
        enc[i * 6 + 2] = ti.sin(freqs[i] * p.y)
        enc[i * 6 + 3] = ti.cos(freqs[i] * p.y)
        enc[i * 6 + 4] = ti.sin(freqs[i] * p.z)
        enc[i * 6 + 5] = ti.cos(freqs[i] * p.z)
    return enc


@ti.kernel
def render_gbuffer():
    for x, y in pos:
        u = (x / RES_X) * 2 - 1
        v = (y / RES_Y) * 2 - 1
        ro = cam_pos[None]
        look = (cam_lookat[None] - ro).normalized()
        right = ti.Vector([0, 1, 0]).cross(look).normalized()
        up = look.cross(right).normalized()
        u_dir = (right * u + up * v * (RES_Y / RES_X) + look).normalized()
        t, id, n = scene.intersect(ro, u_dir)
        if id != -1:
            pos[x, y] = ro + u_dir * t
            normal[x, y] = n
            mat_id[x, y] = id
        else:
            mat_id[x, y] = -1


@ti.kernel
def render_training_rays():
    for i, j in ti.ndrange(RES_X // 4, RES_Y // 4):
        off_x = ti.random() * 4.0
        off_y = ti.random() * 4.0
        x = int(i * 4 + off_x)
        y = int(j * 4 + off_y)
        if x < RES_X and y < RES_Y and mat_id[x, y] != -1:
            p = pos[x, y]
            n = normal[x, y]
            r1 = ti.random()
            r2 = ti.random()
            phi = 2 * math.pi * r1
            sqr_r2 = ti.sqrt(r2)
            local_d = ti.Vector(
                [sqr_r2 * ti.cos(phi), ti.sqrt(1 - r2), sqr_r2 * ti.sin(phi)]
            )
            up = ti.Vector([0, 1, 0]) if abs(n.y) < 0.9 else ti.Vector([1, 0, 0])
            tangent = up.cross(n).normalized()
            bitangent = n.cross(tangent)
            rd = (
                tangent * local_d.x + n * local_d.y + bitangent * local_d.z
            ).normalized()
            t_i, id_i, n_i = scene.intersect(p + n * 1e-3, rd)
            acc_radiance = ti.Vector([0.0, 0.0, 0.0])
            if id_i != -1:
                p_i = p + rd * t_i
                l_vec = light_pos[None] - p_i
                l_dist = l_vec.norm()
                l_dir = l_vec.normalized()
                t_s, id_s, _ = scene.intersect(p_i + n_i * 1e-3, l_dir)
                vis = 1.0 if (id_s == -1 or t_s > l_dist - 0.01) else 0.0
                emission = ti.Vector([0.0, 0.0, 0.0])
                if id_i == 3 and abs(p_i.x) < 0.2 and abs(p_i.z) < 0.2:
                    emission = [15, 15, 15]
                L2 = emission + scene.get_color(id_i) * vis * max(
                    0, n_i.dot(l_dir)
                ) * 25.0 / (l_dist**2 + 0.5)
                acc_radiance = L2 * max(0, n.dot(rd))
            acc_radiance = ti.min(acc_radiance, ti.Vector([10.0, 10.0, 10.0]))
            target_val = ti.log(1.0 + acc_radiance) / 3.0
            idx = ti.atomic_add(splat_ptr[None], 1) % MAX_SPLATS
            s_pos[idx] = p
            s_rad[idx] = target_val
            s_active[idx] = 1


@ti.kernel
def render_inference_and_combine():
    for x, y in neural_out:
        fx, fy = int(x * (RES_X / INTERNAL_X)), int(y * (RES_Y / INTERNAL_Y))
        if mat_id[fx, fy] != -1:
            p = pos[fx, fy]
            n = normal[fx, fy]
            albedo = scene.get_color(mat_id[fx, fy])
            emb = positional_encoding(p)
            h1 = ti.Vector([0.0] * 32)
            for i in range(32):
                val = b1[i]
                for j in range(24):
                    val += W1[i, j] * emb[j]
                h1[i] = max(0.0, val)
            h2 = ti.Vector([0.0] * 32)
            for i in range(32):
                val = b2[i]
                for j in range(32):
                    val += W2[i, j] * h1[j]
                h2[i] = max(0.0, val)
            out = ti.Vector([0.0] * 3)
            for i in range(3):
                val = b3[i]
                for j in range(32):
                    val += W3[i, j] * h2[j]
                out[i] = max(0.0, val)
            indirect = ti.exp(out * 3.0) - 1.0
            direct_acc = ti.Vector([0.0, 0.0, 0.0])
            for s in range(4):
                l_sample = light_pos[None] + ti.Vector(
                    [(ti.random() - 0.5) * 0.4, 0, (ti.random() - 0.5) * 0.4]
                )
                l_vec = l_sample - p
                l_dist = l_vec.norm()
                l_dir = l_vec.normalized()
                t_s, id_s, _ = scene.intersect(p + n * 1e-3, l_dir)
                vis = 1.0 if (id_s == -1 or t_s > l_dist - 0.01) else 0.0
                direct_acc += (
                    albedo * vis * max(0, n.dot(l_dir)) * 25.0 / (l_dist**2 + 0.5)
                )
            direct = direct_acc / 4.0
            color = direct + albedo * indirect
            if mat_id[fx, fy] == 3 and abs(p.x) < 0.2 and abs(p.z) < 0.2:
                color = [15, 15, 15]
            neural_out[x, y] = color
        else:
            neural_out[x, y] = [0, 0, 0]


@ti.func
def world_to_screen_prev(p):
    # REPROJECTION: Project world pos 'p' using PREVIOUS Camera Matrix
    ro = prev_cam_pos[None]
    look = (prev_cam_lookat[None] - ro).normalized()
    right = ti.Vector([0, 1, 0]).cross(look).normalized()
    up = look.cross(right).normalized()

    # FIX: Renamed 'v' to 'rel_pos' to avoid type conflict with scalar 'v' later
    rel_pos = p - ro
    dist = rel_pos.dot(look)
    uv = ti.Vector([-1.0, -1.0])

    if dist > 0.1:
        # Project onto plane
        v_right = rel_pos.dot(right) / dist
        v_up = rel_pos.dot(up) / dist

        # Map back to 0..1
        # FIX: Renamed u/v to u_coord/v_coord to be safe
        u_coord = v_right
        v_coord = v_up / (RES_Y / RES_X)

        uv = ti.Vector([(u_coord + 1) / 2 * RES_X, (v_coord + 1) / 2 * RES_Y])

    return uv


@ti.kernel
def upsample_and_accumulate(dt: ti.f32):
    for x, y in display_image:
        # 1. Bilinear Upscale
        ux = x * (INTERNAL_X / RES_X)
        uy = y * (INTERNAL_Y / RES_Y)
        x0 = int(ux)
        y0 = int(uy)
        x1 = min(x0 + 1, INTERNAL_X - 1)
        y1 = min(y0 + 1, INTERNAL_Y - 1)
        wx = ux - x0
        wy = uy - y0
        c00 = neural_out[x0, y0]
        c10 = neural_out[x1, y0]
        c01 = neural_out[x0, y1]
        c11 = neural_out[x1, y1]
        cur = (c00 * (1 - wx) + c10 * wx) * (1 - wy) + (c01 * (1 - wx) + c11 * wx) * wy

        # 2. TEMPORAL REPROJECTION
        # Get World Pos of current pixel
        p_world = pos[x, y]
        id = mat_id[x, y]

        # History Color
        old_color = ti.Vector([0.0, 0.0, 0.0])
        valid_history = 0.0

        if id != -1:
            # A. Reverse Object Motion
            p_prev = p_world
            if id == 10:  # Box
                p_prev -= box_vel[None] * dt  # Rewind time

            # B. Project into Previous Camera
            uv_prev = world_to_screen_prev(p_prev)

            # C. Sample History
            px = int(uv_prev.x)
            py = int(uv_prev.y)

            if px >= 0 and px < RES_X and py >= 0 and py < RES_Y:
                old_color = prev_image[px, py]
                valid_history = 1.0

        # 3. Blending
        alpha = 0.1  # Standard Blend

        # If history invalid (off screen) or rejected, use current
        if valid_history == 0.0 or frame_idx[None] < 5:
            alpha = 1.0
            old_color = cur

        blended = old_color * (1.0 - alpha) + cur * alpha
        prev_image[x, y] = blended

        # Tone Map
        disp = blended / (blended + 1.0)
        display_image[x, y] = ti.pow(disp, 1.0 / 2.2)


@ti.kernel
def fill_train_data(inp: ti.types.ndarray(), tgt_gi: ti.types.ndarray()):
    for i in range(TRAIN_BATCH):
        r_idx = int(ti.random() * MAX_SPLATS)
        if s_active[r_idx] == 1:
            emb = positional_encoding(s_pos[r_idx])
            for k in range(24):
                inp[i, k] = emb[k]
            val = s_rad[r_idx]
            tgt_gi[i, 0] = val.x
            tgt_gi[i, 1] = val.y
            tgt_gi[i, 2] = val.z


def main():
    model_gi = NeuralCore(out_dim=3, hidden=32).cuda()
    opt_gi = optim.Adam(model_gi.parameters(), lr=0.005)

    in_t = torch.zeros((TRAIN_BATCH, 24), device="cuda")
    tg_gi = torch.zeros((TRAIN_BATCH, 3), device="cuda")

    gui = ti.GUI("Final Neural Renderer + Reprojection", res=(RES_X, RES_Y))
    init_scene()

    f = 0
    paused = False

    while gui.running:
        if gui.get_event(ti.GUI.PRESS):
            if gui.event.key == ti.GUI.SPACE:
                paused = not paused

        # Update Previous State BEFORE input
        prev_cam_pos[None] = cam_pos[None]
        prev_cam_lookat[None] = cam_lookat[None]

        if gui.is_pressed("w"):
            cam_pos[None].z -= 0.05
        if gui.is_pressed("s"):
            cam_pos[None].z += 0.05
        if gui.is_pressed("a"):
            cam_pos[None].x -= 0.05
        if gui.is_pressed("d"):
            cam_pos[None].x += 0.05
        if gui.is_pressed("q"):
            cam_pos[None].y -= 0.05
        if gui.is_pressed("e"):
            cam_pos[None].y += 0.05

        if not paused:
            update_physics(0.016)
        frame_idx[None] = f

        render_gbuffer()
        render_training_rays()

        if f > 0:
            fill_train_data(in_t, tg_gi)
            opt_gi.zero_grad()
            p = model_gi(in_t)
            loss = nn.MSELoss()(p, tg_gi)
            loss.backward()
            opt_gi.step()

            W1.from_numpy(model_gi.net[0].weight.detach().cpu().numpy())
            b1.from_numpy(model_gi.net[0].bias.detach().cpu().numpy())
            W2.from_numpy(model_gi.net[3].weight.detach().cpu().numpy())
            b2.from_numpy(model_gi.net[3].bias.detach().cpu().numpy())
            W3.from_numpy(model_gi.net[6].weight.detach().cpu().numpy())
            b3.from_numpy(model_gi.net[6].bias.detach().cpu().numpy())

        render_inference_and_combine()

        # Pass DT to accumulation for velocity rewind
        upsample_and_accumulate(0.016 if not paused else 0.0)

        gui.set_image(display_image)
        gui.show()
        f += 1


if __name__ == "__main__":
    main()
