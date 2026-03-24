import csv
import math
import os
import time

import numpy as np
import taichi as ti
import torch
import torch.nn as nn
import torch.optim as optim

# --- 1. CONFIGURATION ---
ti.init(arch=ti.cuda, device_memory_fraction=0.9)
RES_X, RES_Y = 1280, 720
INTERNAL_X, INTERNAL_Y = 640, 360  # Render at 50% scale
MAX_SPLATS = 100000
GT_BOUNCES = 3
TRAIN_BATCH = 4096

# --- 2. DATA STRUCTURES ---
pos = ti.Vector.field(3, dtype=ti.f32, shape=(RES_X, RES_Y))
normal = ti.Vector.field(3, dtype=ti.f32, shape=(RES_X, RES_Y))
mat_id = ti.field(dtype=ti.i32, shape=(RES_X, RES_Y))

neural_out = ti.Vector.field(3, dtype=ti.f32, shape=(INTERNAL_X, INTERNAL_Y))
prev_image = ti.Vector.field(3, dtype=ti.f32, shape=(RES_X, RES_Y))
display_image = ti.Vector.field(3, dtype=ti.f32, shape=(RES_X, RES_Y))
gt_image = ti.Vector.field(3, dtype=ti.f32, shape=(RES_X, RES_Y))
last_display_image = ti.Vector.field(3, dtype=ti.f32, shape=(RES_X, RES_Y))

# Ring Buffer (The Cache)
s_pos = ti.Vector.field(3, dtype=ti.f32, shape=MAX_SPLATS)
s_rad = ti.Vector.field(3, dtype=ti.f32, shape=MAX_SPLATS)
splat_ptr = ti.field(dtype=ti.i32, shape=())

cam_pos = ti.Vector.field(3, dtype=ti.f32, shape=())
cam_lookat = ti.Vector.field(3, dtype=ti.f32, shape=())
light_pos = ti.Vector.field(3, dtype=ti.f32, shape=())
box_pos = ti.Vector.field(3, dtype=ti.f32, shape=())
frame_idx = ti.field(dtype=ti.i32, shape=())


# --- 3. NEURAL NETWORK (Optimized 32-Neuron MLP) ---
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
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x)


W1 = ti.field(dtype=ti.f32, shape=(32, 24))
b1 = ti.field(dtype=ti.f32, shape=(32))
W2 = ti.field(dtype=ti.f32, shape=(32, 32))
b2 = ti.field(dtype=ti.f32, shape=(32))
W3 = ti.field(dtype=ti.f32, shape=(3, 32))
b3 = ti.field(dtype=ti.f32, shape=(3))


# --- 4. SCENE ---
@ti.data_oriented
class SceneGeometry:
    def __init__(self):
        self.light_size = 0.4

    @ti.func
    def sample_light(self):
        rx = (ti.random() - 0.5) * self.light_size
        rz = (ti.random() - 0.5) * self.light_size
        return light_pos[None] + ti.Vector([rx, 0.0, rz])

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


# --- 5. KERNELS ---
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
        f = (cam_lookat[None] - ro).normalized()
        r = ti.Vector([0, 1, 0]).cross(f).normalized()
        u_dir = (r * u + ti.Vector([0, 1, 0]) * v * (RES_Y / RES_X) + f).normalized()
        t, id, n = scene.intersect(ro, u_dir)
        if id != -1:
            pos[x, y] = ro + u_dir * t
            normal[x, y] = n
            mat_id[x, y] = id
        else:
            mat_id[x, y] = -1


@ti.kernel
def render_baseline_lumen_like():
    for bx, by in ti.ndrange(RES_X // 4, RES_Y // 4):
        cx, cy = bx * 4 + 2, by * 4 + 2
        radiance = ti.Vector([0.0, 0.0, 0.0])
        if mat_id[cx, cy] != -1:
            p = pos[cx, cy]
            n = normal[cx, cy]
            albedo = scene.get_color(mat_id[cx, cy])
            l_vec = light_pos[None] - p
            l_dist = l_vec.norm()
            l_dir = l_vec.normalized()
            t_s, id_s, _ = scene.intersect(p + n * 1e-3, l_dir)
            vis = 1.0 if (id_s == -1 or t_s > l_dist - 0.01) else 0.0
            direct = albedo * vis * max(0, n.dot(l_dir)) * 25.0 / (l_dist**2 + 0.5)

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
            indirect = ti.Vector([0.0, 0.0, 0.0])
            if id_i != -1:
                p_i = p + rd * t_i
                l_vec_i = light_pos[None] - p_i
                l_dist_i = l_vec_i.norm()
                l_dir_i = l_vec_i.normalized()
                t_si, id_si, _ = scene.intersect(p_i + n_i * 1e-3, l_dir_i)
                vis_i = 1.0 if (id_si == -1 or t_si > l_dist_i - 0.01) else 0.0
                indirect = (
                    scene.get_color(id_i)
                    * vis_i
                    * max(0, n_i.dot(l_dir_i))
                    * 25.0
                    / (l_dist_i**2 + 0.5)
                )
            radiance = direct + albedo * indirect
            if mat_id[cx, cy] == 3 and abs(p.x) < 0.2 and abs(p.z) < 0.2:
                radiance = [15, 15, 15]

        for i, j in ti.static(ti.ndrange(4, 4)):
            px, py = bx * 4 + i, by * 4 + j
            if px < RES_X and py < RES_Y:
                display_image[px, py] = radiance

    for x, y in display_image:
        blended = prev_image[x, y] * 0.92 + display_image[x, y] * 0.08
        if frame_idx[None] < 2:
            blended = display_image[x, y]
        prev_image[x, y] = blended
        display_image[x, y] = blended


@ti.kernel
def render_neural_training_rays():
    # Dense Sampling (1 ray per 16 pixels)
    for i, j in ti.ndrange(RES_X // 4, RES_Y // 4):
        # Spatial Jitter: Cover different pixels every frame!
        off_x = ti.random() * 4.0
        off_y = ti.random() * 4.0
        x = int(i * 4 + off_x)
        y = int(j * 4 + off_y)

        if x < RES_X and y < RES_Y and mat_id[x, y] != -1:
            p = pos[x, y]
            n = normal[x, y]

            # Trace Indirect
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
                l_target_b = scene.sample_light()
                l_vec_b = l_target_b - p_i
                l_dist_b = l_vec_b.norm()
                l_dir_b = l_vec_b.normalized()
                t_s_b, id_s_b, _ = scene.intersect(p_i + n_i * 1e-3, l_dir_b)
                vis_b = 1.0 if (id_s_b == -1 or t_s_b > l_dist_b - 0.01) else 0.0
                L2 = (
                    scene.get_color(id_i)
                    * vis_b
                    * max(0, n_i.dot(l_dir_b))
                    * 25.0
                    / (l_dist_b**2 + 0.5)
                )
                acc_radiance = L2 * max(0, n.dot(rd))

            acc_radiance = ti.min(acc_radiance, ti.Vector([15.0, 15.0, 15.0]))
            target_val = ti.log(1.0 + acc_radiance) / 3.0

            # --- THE FIX: NO BLENDING. PURE OVERWRITE. ---
            idx = ti.atomic_add(splat_ptr[None], 1) % MAX_SPLATS
            s_pos[idx] = p
            s_rad[idx] = target_val


@ti.kernel
def render_neural_inference_lowres():
    for x, y in neural_out:
        fx, fy = int(x * (RES_X / INTERNAL_X)), int(y * (RES_Y / INTERNAL_Y))
        if mat_id[fx, fy] != -1:
            p = pos[fx, fy]
            n = normal[fx, fy]
            albedo = scene.get_color(mat_id[fx, fy])
            emb = positional_encoding(p)

            # 32-Neuron Inference
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
                out[i] = 1.0 / (1.0 + ti.exp(-val))
            indirect = ti.exp(out * 3.0) - 1.0

            # Analytic Direct
            direct_acc = ti.Vector([0.0, 0.0, 0.0])
            for s in range(4):
                l_target = scene.sample_light()
                l_vec = l_target - p
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


@ti.kernel
def upsample_and_accumulate():
    for x, y in display_image:
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

        old = prev_image[x, y]
        alpha = 0.08
        if frame_idx[None] < 2:
            alpha = 1.0
        blended = old * (1.0 - alpha) + cur * alpha
        prev_image[x, y] = blended
        display_image[x, y] = blended


@ti.kernel
def render_ground_truth(accum_frame: ti.i32):
    for x, y in gt_image:
        u = (x + ti.random()) / RES_X * 2 - 1
        v = (y + ti.random()) / RES_Y * 2 - 1
        ro = cam_pos[None]
        f = (cam_lookat[None] - ro).normalized()
        r = ti.Vector([0, 1, 0]).cross(f).normalized()
        u_dir = (r * u + ti.Vector([0, 1, 0]) * v * (RES_Y / RES_X) + f).normalized()
        throughput = ti.Vector([1.0, 1.0, 1.0])
        radiance = ti.Vector([0.0, 0.0, 0.0])
        cur_ro = ro
        cur_rd = u_dir
        for depth in range(GT_BOUNCES):
            t, id, n = scene.intersect(cur_ro, cur_rd)
            if id == -1:
                break
            p = cur_ro + cur_rd * t
            albedo = scene.get_color(id)
            if id == 3 and abs(p.x) < 0.2 and abs(p.z) < 0.2:
                radiance += throughput * 15.0
                break
            l_target = scene.sample_light()
            l_vec = l_target - p
            l_dist = l_vec.norm()
            l_dir = l_vec.normalized()
            t_s, id_s, _ = scene.intersect(p + n * 1e-3, l_dir)
            vis = 1.0 if (id_s == -1 or t_s > l_dist - 0.01) else 0.0
            direct = albedo * vis * max(0, n.dot(l_dir)) * 25.0 / (l_dist**2 + 0.5)
            radiance += throughput * direct
            throughput *= albedo
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
            cur_rd = (
                tangent * local_d.x + n * local_d.y + bitangent * local_d.z
            ).normalized()
            cur_ro = p + n * 1e-3
        if accum_frame == 0:
            gt_image[x, y] = radiance
        else:
            gt_image[x, y] = (gt_image[x, y] * accum_frame + radiance) / (
                accum_frame + 1
            )


# --- 6. METRICS ---
@ti.kernel
def backup_last_frame():
    for x, y in display_image:
        last_display_image[x, y] = display_image[x, y]


@ti.kernel
def compute_metrics(tgt: ti.template(), ref: ti.template()) -> ti.types.vector(
    3, ti.f32
):
    total_mse = 0.0
    total_temp = 0.0
    shadow_sum = 0.0
    shadow_sq_sum = 0.0
    shadow_count = 0.0
    for x, y in tgt:
        diff = tgt[x, y] - ref[x, y]
        total_mse += diff.dot(diff)
        tdiff = tgt[x, y] - last_display_image[x, y]
        total_temp += tdiff.norm()
        if 540 < x < 740 and 260 < y < 460:
            lum = tgt[x, y].norm()
            shadow_sum += lum
            shadow_sq_sum += lum * lum
            shadow_count += 1.0
    mse = total_mse / (RES_X * RES_Y)
    temp = total_temp / (RES_X * RES_Y)
    s_mean = shadow_sum / shadow_count
    s_var = (shadow_sq_sum / shadow_count) - (s_mean * s_mean)
    return ti.Vector([mse, temp, ti.sqrt(s_var)])


@ti.kernel
def fill_train_data(inp: ti.types.ndarray(), tgt_gi: ti.types.ndarray()):
    # RANDOM SAMPLING from Buffer (Monte Carlo Learning)
    for i in range(TRAIN_BATCH):
        r_idx = int(ti.random() * MAX_SPLATS)
        emb = positional_encoding(s_pos[r_idx])
        for k in range(24):
            inp[i, k] = emb[k]
        val = s_rad[r_idx]
        tgt_gi[i, 0] = val.x
        tgt_gi[i, 1] = val.y
        tgt_gi[i, 2] = val.z


def run_scene(scene_name, mode, writer):
    print(f"--- {scene_name} : {mode} ---")
    cam_pos[None] = [0, 0, 3.5]
    cam_lookat[None] = [0, 0, 0]
    light_pos[None] = [0, 1.9, 0]
    box_pos[None] = [0, -0.6, 0]
    frame_idx[None] = 0
    splat_ptr[None] = 0
    prev_image.fill([0, 0, 0])
    last_display_image.fill([0, 0, 0])

    # 32 Neurons = Fast
    model_gi = NeuralCore(out_dim=3, hidden=32).cuda()
    opt_gi = optim.Adam(model_gi.parameters(), lr=0.005)

    in_t = torch.zeros((TRAIN_BATCH, 24), device="cuda")
    tg_gi = torch.zeros((TRAIN_BATCH, 3), device="cuda")

    log = []
    frames = 120 if scene_name != "SCENE_A" else 400

    for f in range(frames):
        frame_idx[None] = f
        if scene_name == "SCENE_A":
            light_pos[None] = [0.6, 1.9, 0] if f >= 200 else [0, 1.9, 0]
        elif scene_name == "SCENE_B":
            box_pos[None] = [-1.0 + (f / 120.0) * 2.0, -0.6, 0]
        elif scene_name == "SCENE_C":
            a = (f / 120.0) * 2 * math.pi
            cam_pos[None] = [math.sin(a) * 3.5, 0, math.cos(a) * 3.5]

        t0 = time.time()

        # CLEAR BUFFER TO AVOID POISONING
        # s_pos.fill(0) # Not strictly needed if we randomly sample valid indices, but safer

        render_gbuffer()

        if mode == "NEURAL":
            render_neural_training_rays()

            # Train Once Per Frame
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

            render_neural_inference_lowres()
            upsample_and_accumulate()

        elif mode == "BASELINE":
            render_baseline_lumen_like()
        elif mode == "GT":
            render_ground_truth(f)
            continue

        ti.sync()
        dt = (time.time() - t0) * 1000.0

        m = compute_metrics(display_image, gt_image)
        rmse = math.sqrt(m[0])
        psnr = 20 * math.log10(1.0 / rmse) if rmse > 0 else 0

        backup_last_frame()
        writer.writerow([f, dt, 1000 / max(dt, 0.01), rmse, psnr, m[1], m[2]])
        log.append([f, dt, 0, rmse, psnr, m[1], m[2]])
    return log


if __name__ == "__main__":
    scenes = ["SCENE_A", "SCENE_B", "SCENE_C"]
    modes = ["BASELINE", "NEURAL"]
    results = {}
    for s in scenes:
        print(f"Generating GT for {s}...")
        gt_image.fill([0, 0, 0])
        run_scene(s, "GT", csv.writer(open(os.devnull, "w")))
        for m in modes:
            with open(f"bench_{s}_{m}.csv", "w", newline="") as f:
                log = run_scene(s, m, csv.writer(f))
                avg_ms = np.mean([r[1] for r in log])
                avg_psnr = np.mean([r[4] for r in log])
                avg_temp = np.mean([r[5] for r in log])
                results[f"{s}_{m}"] = {"ms": avg_ms, "psnr": avg_psnr, "temp": avg_temp}

    print("\n=== FINAL REPORT ===")
    print(f"{'Metric':<20} | {'Base':<8} | {'Neur':<8} | {'Res'}")
    b_ms = np.mean([results[f"{s}_BASELINE"]["ms"] for s in scenes])
    n_ms = np.mean([results[f"{s}_NEURAL"]["ms"] for s in scenes])
    print(
        f"{'Frame Time':<20} | {b_ms:.2f}ms | {n_ms:.2f}ms | {'PASS' if n_ms <= 16.6 else 'FAIL'}"
    )
    b_psnr = np.mean([results[f"{s}_BASELINE"]["psnr"] for s in scenes])
    n_psnr = np.mean([results[f"{s}_NEURAL"]["psnr"] for s in scenes])
    print(
        f"{'PSNR':<20} | {b_psnr:.2f}dB | {n_psnr:.2f}dB | {'PASS' if n_psnr >= b_psnr - 2 else 'FAIL'}"
    )
    b_stab = results["SCENE_C_BASELINE"]["temp"]
    n_stab = results["SCENE_C_NEURAL"]["temp"]
    print(
        f"{'Stability':<20} | {b_stab:.4f}   | {n_stab:.4f}   | {'PASS' if n_stab < b_stab * 0.7 else 'FAIL'}"
    )
