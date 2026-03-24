import math

import numpy as np
import taichi as ti
import torch
import torch.nn as nn
import torch.optim as optim

ti.init(arch=ti.cuda)
RES_X, RES_Y = 640, 360
MAX_SPLATS = 100000
TRAIN_BATCH = 8192

# --- PHYSICS ---
box_pos = ti.Vector.field(3, dtype=ti.f32, shape=())
box_vel = ti.Vector.field(3, dtype=ti.f32, shape=())

# --- G-BUFFER ---
pos = ti.Vector.field(3, dtype=ti.f32, shape=(RES_X, RES_Y))
normal = ti.Vector.field(3, dtype=ti.f32, shape=(RES_X, RES_Y))
mat_id = ti.field(dtype=ti.i32, shape=(RES_X, RES_Y))

# --- HISTORY BUFFERS ---
raw_image = ti.Vector.field(3, dtype=ti.f32, shape=(RES_X, RES_Y))
prev_image = ti.Vector.field(3, dtype=ti.f32, shape=(RES_X, RES_Y))
final_image = ti.Vector.field(3, dtype=ti.f32, shape=(RES_X, RES_Y))

s_pos = ti.Vector.field(3, dtype=ti.f32, shape=MAX_SPLATS)
s_rad = ti.Vector.field(3, dtype=ti.f32, shape=MAX_SPLATS)
s_active = ti.field(dtype=ti.i32, shape=MAX_SPLATS)
splat_count = ti.field(dtype=ti.i32, shape=())

cam_pos = ti.Vector.field(3, dtype=ti.f32, shape=())
frame_idx = ti.field(dtype=ti.i32, shape=())
time_val = ti.field(dtype=ti.f32, shape=())

W1 = ti.field(dtype=ti.f32, shape=(64, 24))
b1 = ti.field(dtype=ti.f32, shape=(64))
W2 = ti.field(dtype=ti.f32, shape=(64, 64))
b2 = ti.field(dtype=ti.f32, shape=(64))
W3 = ti.field(dtype=ti.f32, shape=(3, 64))
b3 = ti.field(dtype=ti.f32, shape=(3))


# --- PYTORCH MODEL ---
class NeuralCache(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(24, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Linear(64, 3),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x)


# --- PHYSICS KERNELS ---
@ti.kernel
def init_physics():
    box_pos[None] = [0.0, -0.5, 0.0]
    box_vel[None] = [1.5, 0.8, -0.8]


@ti.kernel
def update_physics(dt: ti.f32):
    box_pos[None] += box_vel[None] * dt
    b = 0.55
    if box_pos[None].x > b:
        box_pos[None].x = b
        box_vel[None].x *= -1.0
    if box_pos[None].x < -b:
        box_pos[None].x = -b
        box_vel[None].x *= -1.0
    if box_pos[None].y > b:
        box_pos[None].y = b
        box_vel[None].y *= -1.0
    if box_pos[None].y < -b:
        box_pos[None].y = -b
        box_vel[None].y *= -1.0
    if box_pos[None].z > 3.0:
        box_pos[None].z = 3.0
        box_vel[None].z *= -1.0
    if box_pos[None].z < -b:
        box_pos[None].z = -b
        box_vel[None].z *= -1.0

    if ti.random() < 0.03:
        impulse = (
            ti.Vector([ti.random() - 0.5, ti.random() - 0.5, ti.random() - 0.5]) * 3.0
        )
        box_vel[None] += impulse
        if box_vel[None].norm() > 4.0:
            box_vel[None] = box_vel[None].normalized() * 4.0


# --- GEOMETRY ---
@ti.data_oriented
class Scene:
    def __init__(self):
        self.light_center = ti.Vector.field(3, dtype=ti.f32, shape=())
        self.light_center[None] = [0.0, 0.95, 0.0]
        self.light_size = 0.4

    @ti.func
    def sample_light(self):
        rx = (ti.random() - 0.5) * self.light_size
        rz = (ti.random() - 0.5) * self.light_size
        return self.light_center[None] + ti.Vector([rx, 0.0, rz])

    @ti.func
    def intersect_aabb(self, ro, rd, box_min, box_max):
        t_min = -1e9
        t_max = 1e9
        hit = True
        for i in ti.static(range(3)):
            if abs(rd[i]) < 1e-6:
                if ro[i] < box_min[i] or ro[i] > box_max[i]:
                    hit = False
            else:
                inv_d = 1.0 / rd[i]
                t1 = (box_min[i] - ro[i]) * inv_d
                t2 = (box_max[i] - ro[i]) * inv_d
                t_min = max(t_min, min(t1, t2))
                t_max = min(t_max, max(t1, t2))
        if t_max <= t_min or t_max <= 0.0:
            hit = False
        return hit, t_min

    @ti.func
    def get_normal_aabb(self, p, b_min, b_max):
        n = ti.Vector([0.0, 1.0, 0.0])
        eps = 1e-3
        if abs(p.x - b_min.x) < eps:
            n = [-1, 0, 0]
        elif abs(p.x - b_max.x) < eps:
            n = [1, 0, 0]
        elif abs(p.y - b_min.y) < eps:
            n = [0, -1, 0]
        elif abs(p.y - b_max.y) < eps:
            n = [0, 1, 0]
        elif abs(p.z - b_min.z) < eps:
            n = [0, 0, -1]
        elif abs(p.z - b_max.z) < eps:
            n = [0, 0, 1]
        return n

    @ti.func
    def intersect_scene(self, ro, rd):
        t_near = 1e9
        id_near = -1
        n_near = ti.Vector([0.0, 0.0, 0.0])
        x_min, x_max = -1.0, 1.0
        y_min, y_max = -1.0, 1.0
        z_min, z_max = -1.0, 4.0

        t = (x_min - ro.x) / rd.x
        p = ro + rd * t
        if t > 0.001 and t < t_near and y_min <= p.y <= y_max and z_min <= p.z <= z_max:
            t_near = t
            id_near = 0
            n_near = [1, 0, 0]
        t = (x_max - ro.x) / rd.x
        p = ro + rd * t
        if t > 0.001 and t < t_near and y_min <= p.y <= y_max and z_min <= p.z <= z_max:
            t_near = t
            id_near = 1
            n_near = [-1, 0, 0]
        t = (y_min - ro.y) / rd.y
        p = ro + rd * t
        if t > 0.001 and t < t_near and x_min <= p.x <= x_max and z_min <= p.z <= z_max:
            t_near = t
            id_near = 2
            n_near = [0, 1, 0]
        t = (y_max - ro.y) / rd.y
        p = ro + rd * t
        if t > 0.001 and t < t_near and x_min <= p.x <= x_max and z_min <= p.z <= z_max:
            t_near = t
            id_near = 3
            n_near = [0, -1, 0]
        t = (z_min - ro.z) / rd.z
        p = ro + rd * t
        if t > 0.001 and t < t_near and x_min <= p.x <= x_max and y_min <= p.y <= y_max:
            t_near = t
            id_near = 4
            n_near = [0, 0, 1]
        t = (z_max - ro.z) / rd.z
        p = ro + rd * t
        if t > 0.001 and t < t_near and x_min <= p.x <= x_max and y_min <= p.y <= y_max:
            t_near = t
            id_near = 5
            n_near = [0, 0, -1]

        c = box_pos[None]
        b_min = c - 0.4
        b_max = c + 0.4
        hit, t_box = self.intersect_aabb(ro, rd, b_min, b_max)
        if hit and t_box < t_near and t_box > 0.001:
            t_near = t_box
            id_near = 10
            n_near = self.get_normal_aabb(ro + rd * t_box, b_min, b_max)
        return t_near, id_near, n_near

    @ti.func
    def get_color(self, id):
        c = ti.Vector([0.0, 0.0, 0.0])
        if id == 0:
            c = [0.8, 0.1, 0.1]
        elif id == 1:
            c = [0.1, 0.8, 0.1]
        elif id == 2:
            c = [0.9, 0.9, 0.9]
        elif id == 5:
            c = [0.9, 0.9, 0.9]
        elif id == 10:
            c = [0.2, 0.4, 0.9]
        else:
            c = [0.8, 0.8, 0.8]
        return c


scene = Scene()


# --- KERNELS ---
@ti.kernel
def render_gbuffer():
    for x, y in pos:
        u = (x / RES_X) * 2 - 1
        v = (y / RES_Y) * 2 - 1
        ro = cam_pos[None]
        f = (ti.Vector([0, 0, 0]) - ro).normalized()
        r = ti.Vector([0, 1, 0]).cross(f).normalized()
        u_dir = (r * u + ti.Vector([0, 1, 0]) * v * (RES_Y / RES_X) + f).normalized()
        t, id, n = scene.intersect_scene(ro, u_dir)
        if id != -1:
            pos[x, y] = ro + u_dir * t
            normal[x, y] = n
            mat_id[x, y] = id
        else:
            mat_id[x, y] = -1


@ti.kernel
def trace_and_splat():
    for i, j in ti.ndrange(RES_X // 4, RES_Y // 4):
        x, y = i * 4, j * 4
        if mat_id[x, y] != -1:
            p_0 = pos[x, y]
            n_0 = normal[x, y]
            r1 = ti.random()
            r2 = ti.random()
            phi = 2 * 3.14159 * r1
            sqr_r2 = ti.sqrt(r2)
            local_d = ti.Vector(
                [sqr_r2 * ti.cos(phi), ti.sqrt(1 - r2), sqr_r2 * ti.sin(phi)]
            )
            up = ti.Vector([0, 1, 0])
            if abs(n_0.y) > 0.9:
                up = ti.Vector([1, 0, 0])
            tangent = up.cross(n_0).normalized()
            bitangent = n_0.cross(tangent)
            rd_1 = (
                tangent * local_d.x + n_0 * local_d.y + bitangent * local_d.z
            ).normalized()
            t_1, id_1, n_1 = scene.intersect_scene(p_0 + n_0 * 1e-3, rd_1)
            acc_radiance = ti.Vector([0.0, 0.0, 0.0])
            if id_1 != -1:
                p_1 = p_0 + rd_1 * t_1
                albedo_1 = scene.get_color(id_1)
                l_target = scene.sample_light()
                l_vec = l_target - p_1
                l_dist = l_vec.norm()
                l_dir = l_vec.normalized()
                t_s, id_s, _ = scene.intersect_scene(p_1 + n_1 * 1e-3, l_dir)
                vis = 1.0 if (id_s == -1 or t_s > l_dist - 0.01) else 0.0
                L2 = albedo_1 * vis * max(0, n_1.dot(l_dir)) * 25.0 / (l_dist**2 + 0.5)
                acc_radiance = L2 * max(0, n_1.dot(-rd_1))

            acc_radiance = ti.min(acc_radiance, ti.Vector([15.0, 15.0, 15.0]))
            idx = ti.atomic_add(splat_count[None], 1) % MAX_SPLATS
            if s_active[idx] == 1:
                s_rad[idx] = s_rad[idx] * 0.9 + acc_radiance * 0.1
                s_pos[idx] = p_0
            else:
                s_active[idx] = 1
                s_pos[idx] = p_0
                s_rad[idx] = acc_radiance


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
def neural_inference():
    for x, y in raw_image:
        if mat_id[x, y] == -1:
            raw_image[x, y] = [0, 0, 0]
        else:
            p = pos[x, y]
            n = normal[x, y]
            albedo = scene.get_color(mat_id[x, y])
            emb = positional_encoding(p)
            h1 = ti.Vector([0.0] * 64)
            for i in range(64):
                val = b1[i]
                for j in range(24):
                    val += W1[i, j] * emb[j]
                h1[i] = max(0.0, val)
            h2 = ti.Vector([0.0] * 64)
            for i in range(64):
                val = b2[i]
                for j in range(64):
                    val += W2[i, j] * h1[j]
                h2[i] = max(0.0, val)
            out = ti.Vector([0.0] * 3)
            for i in range(3):
                val = b3[i]
                for j in range(64):
                    val += W3[i, j] * h2[j]
                out[i] = 1.0 / (1.0 + ti.exp(-val))
            indirect_log = out * 3.0
            indirect = ti.exp(indirect_log) - 1.0

            # Direct Light (16 Samples)
            direct_acc = ti.Vector([0.0, 0.0, 0.0])
            for s in range(16):
                l_target = scene.sample_light()
                l_vec = l_target - p
                l_dist = l_vec.norm()
                l_dir = l_vec.normalized()
                t_s, id_s, _ = scene.intersect_scene(p + n * 1e-3, l_dir)
                vis = 1.0 if (id_s == -1 or t_s > l_dist - 0.01) else 0.0
                direct_acc += (
                    albedo * vis * max(0, n.dot(l_dir)) * 30.0 / (l_dist**2 + 0.5)
                )
            direct = direct_acc / 16.0

            color = direct + (albedo * indirect)
            if mat_id[x, y] == 3 and abs(p.x) < 0.5 and abs(p.z) < 0.5:
                color = [15, 15, 15]
            raw_image[x, y] = color


@ti.kernel
def temporal_accumulate():
    for x, y in final_image:
        cur = raw_image[x, y]
        old = prev_image[x, y]

        # --- FIX: NEIGHBORHOOD CLAMPING (The TAA Magic) ---
        # Find Min/Max of current 3x3 neighborhood
        c_min = cur
        c_max = cur
        for i in range(-1, 2):
            for j in range(-1, 2):
                nx, ny = x + i, y + j
                if 0 <= nx < RES_X and 0 <= ny < RES_Y:
                    val = raw_image[nx, ny]
                    c_min = ti.min(c_min, val)
                    c_max = ti.max(c_max, val)

        # Clamp History to valid range -> Kills Ghosting!
        old_clamped = ti.min(ti.max(old, c_min), c_max)

        # Blend
        alpha = 0.1  # Very smooth blend
        if frame_idx[None] < 2:
            alpha = 1.0

        blended = old_clamped * (1.0 - alpha) + cur * alpha
        prev_image[x, y] = blended

        c_disp = blended / (blended + 1.0)
        final_image[x, y] = ti.pow(c_disp, 1.0 / 2.2)


@ti.kernel
def fill_train_data(inp: ti.types.ndarray(), tgt: ti.types.ndarray()):
    count = 0
    for i in range(MAX_SPLATS):
        if s_active[i]:
            idx = ti.atomic_add(count, 1)
            if idx < TRAIN_BATCH:
                emb = positional_encoding(s_pos[i])
                for k in range(24):
                    inp[idx, k] = emb[k]
                val = s_rad[i]
                tgt[idx, 0] = ti.log(1.0 + val.x) / 3.0
                tgt[idx, 1] = ti.log(1.0 + val.y) / 3.0
                tgt[idx, 2] = ti.log(1.0 + val.z) / 3.0


def main():
    model = NeuralCache().cuda()
    optimizer = optim.Adam(model.parameters(), lr=0.005)
    gui = ti.GUI("Polished Neural Oracle", res=(RES_X, RES_Y))

    in_tensor = torch.zeros((TRAIN_BATCH, 24), device="cuda")
    tg_tensor = torch.zeros((TRAIN_BATCH, 3), device="cuda")

    init_physics()

    f_idx = 0
    while gui.running:
        f_idx += 1
        frame_idx[None] = f_idx
        cam_pos[None] = [0, 0, 3.5]

        update_physics(0.016)

        render_gbuffer()
        trace_and_splat()
        fill_train_data(in_tensor, tg_tensor)

        for _ in range(3):
            optimizer.zero_grad()
            pred = model(in_tensor)
            loss = nn.MSELoss()(pred, tg_tensor)
            loss.backward()
            optimizer.step()

        W1.from_numpy(model.net[0].weight.detach().cpu().numpy())
        b1.from_numpy(model.net[0].bias.detach().cpu().numpy())
        W2.from_numpy(model.net[3].weight.detach().cpu().numpy())
        b2.from_numpy(model.net[3].bias.detach().cpu().numpy())
        W3.from_numpy(model.net[6].weight.detach().cpu().numpy())
        b3.from_numpy(model.net[6].bias.detach().cpu().numpy())

        neural_inference()
        temporal_accumulate()
        gui.set_image(final_image)
        gui.show()


if __name__ == "__main__":
    main()
