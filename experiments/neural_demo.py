import math
import time

import numpy as np
import taichi as ti
import torch
import torch.nn as nn
import torch.optim as optim

# ------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------
ti.init(arch=ti.cuda, device_memory_fraction=0.8)

RES_X, RES_Y = 640, 360  # render resolution (lowered so you can actually run)
INTERNAL_X, INTERNAL_Y = 320, 180
MAX_SPLATS = 50000
TRAIN_BATCH = 2048

# ------------------------------------------------------------
# BUFFERS
# ------------------------------------------------------------
pos = ti.Vector.field(3, dtype=ti.f32, shape=(RES_X, RES_Y))
normal = ti.Vector.field(3, dtype=ti.f32, shape=(RES_X, RES_Y))
mat_id = ti.field(dtype=ti.i32, shape=(RES_X, RES_Y))

neural_out = ti.Vector.field(3, dtype=ti.f32, shape=(INTERNAL_X, INTERNAL_Y))
prev_image = ti.Vector.field(3, dtype=ti.f32, shape=(RES_X, RES_Y))
display_image = ti.Vector.field(3, dtype=ti.f32, shape=(RES_X, RES_Y))

# splat buffer (your idea)
s_pos = ti.Vector.field(3, dtype=ti.f32, shape=MAX_SPLATS)
s_rad = ti.Vector.field(3, dtype=ti.f32, shape=MAX_SPLATS)
splat_ptr = ti.field(dtype=ti.i32, shape=())

# scene state
cam_pos = ti.Vector.field(3, dtype=ti.f32, shape=())
cam_look = ti.Vector.field(3, dtype=ti.f32, shape=())
light_pos = ti.Vector.field(3, dtype=ti.f32, shape=())
box_pos = ti.Vector.field(3, dtype=ti.f32, shape=())
frame_idx = ti.field(dtype=ti.i32, shape=())


# ------------------------------------------------------------
# NEURAL CORE (your architecture, just cleaned)
# ------------------------------------------------------------
class NeuralCore(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(24, 32),
            nn.LayerNorm(32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.LayerNorm(32),
            nn.ReLU(),
            nn.Linear(32, 3),  # radiance prediction
        )

        # tiny init to avoid flashbangs
        nn.init.uniform_(self.net[-1].weight, -0.001, 0.001)
        nn.init.constant_(self.net[-1].bias, 0.0)

    def forward(self, x):
        return self.net(x)


# Taichi copies of weights
W1 = ti.field(dtype=ti.f32, shape=(32, 24))
b1 = ti.field(dtype=ti.f32, shape=(32))
W2 = ti.field(dtype=ti.f32, shape=(32, 32))
b2 = ti.field(dtype=ti.f32, shape=(32))
W3 = ti.field(dtype=ti.f32, shape=(3, 32))
b3 = ti.field(dtype=ti.f32, shape=(3))


# ------------------------------------------------------------
# SCENE
# ------------------------------------------------------------
@ti.data_oriented
class Scene:
    @ti.func
    def intersect(self, ro, rd):
        t_min = 1e9
        id_min = -1
        n_min = ti.Vector([0.0, 0.0, 0.0])

        # left wall
        t = (-1.0 - ro.x) / rd.x
        p = ro + rd * t
        if t > 0.001 and abs(p.y) < 1 and abs(p.z) < 1:
            t_min = t
            id_min = 0
            n_min = [1, 0, 0]

        # right wall
        t = (1.0 - ro.x) / rd.x
        p = ro + rd * t
        if t > 0.001 and t < t_min and abs(p.y) < 1 and abs(p.z) < 1:
            t_min = t
            id_min = 1
            n_min = [-1, 0, 0]

        # floor
        t = (-1.0 - ro.y) / rd.y
        p = ro + rd * t
        if t > 0.001 and t < t_min and abs(p.x) < 1 and abs(p.z) < 1:
            t_min = t
            id_min = 2
            n_min = [0, 1, 0]

        # ceiling
        t = (1.0 - ro.y) / rd.y
        p = ro + rd * t
        if t > 0.001 and t < t_min and abs(p.x) < 1 and abs(p.z) < 1:
            t_min = t
            id_min = 3
            n_min = [0, -1, 0]

        # back wall
        t = (-2.0 - ro.z) / rd.z
        p = ro + rd * t
        if t > 0.001 and t < t_min and abs(p.x) < 1 and abs(p.y) < 1:
            t_min = t
            id_min = 4
            n_min = [0, 0, 1]

        # moving box
        c = box_pos[None]
        bmin = c - 0.4
        bmax = c + 0.4

        t_box = 1e9
        for i in ti.static(range(3)):
            if abs(rd[i]) > 1e-6:
                inv = 1.0 / rd[i]
                t1 = (bmin[i] - ro[i]) * inv
                t2 = (bmax[i] - ro[i]) * inv
                t_box = min(t_box, max(t1, t2))

        if t_box > 0.001 and t_box < t_min:
            t_min = t_box
            id_min = 10
            p_hit = ro + rd * t_box
            n_min = ti.Vector([0, 1, 0])

        return t_min, id_min, n_min

    @ti.func
    def get_color(self, id):
        # Start with default (white walls)
        c = ti.Vector([0.9, 0.9, 0.9])

        if id == 0:  # red wall
            c = ti.Vector([0.8, 0.1, 0.1])
        elif id == 1:  # green wall
            c = ti.Vector([0.1, 0.8, 0.1])
        elif id == 10:  # blue box
            c = ti.Vector([0.2, 0.4, 0.9])

        return c


scene = Scene()


# ------------------------------------------------------------
# POSITIONAL ENCODING
# ------------------------------------------------------------
@ti.func
def pe(p):
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


# ------------------------------------------------------------
# GBUFFER
# ------------------------------------------------------------
@ti.kernel
def render_gbuffer():
    for x, y in pos:
        u = (x / RES_X) * 2 - 1
        v = (y / RES_Y) * 2 - 1
        ro = cam_pos[None]
        look = (cam_look[None] - ro).normalized()
        right = ti.Vector([0, 1, 0]).cross(look).normalized()
        up = look.cross(right).normalized()
        rd = (right * u + up * v + look).normalized()

        t, id, n = scene.intersect(ro, rd)
        if id != -1:
            pos[x, y] = ro + rd * t
            normal[x, y] = n
            mat_id[x, y] = id
        else:
            mat_id[x, y] = -1


# ------------------------------------------------------------
# TRAINING RAYS (fill splat buffer)
# ------------------------------------------------------------
@ti.kernel
def render_training_rays():
    for i, j in ti.ndrange(RES_X // 4, RES_Y // 4):
        x = i * 4
        y = j * 4
        if mat_id[x, y] != -1:
            p = pos[x, y]
            n = normal[x, y]

            # simple shadow probe
            l = (light_pos[None] - p).normalized()
            t_s, id_s, _ = scene.intersect(p + n * 1e-3, l)
            vis = 1.0 if id_s == -1 else 0.0

            albedo = scene.get_color(mat_id[x, y])
            radiance = albedo * vis * 5.0

            idx = ti.atomic_add(splat_ptr[None], 1) % MAX_SPLATS
            s_pos[idx] = p
            s_rad[idx] = radiance


# ------------------------------------------------------------
# FILL TRAIN DATA FOR PYTORCH
# ------------------------------------------------------------
@ti.kernel
def fill_train_data(inp: ti.types.ndarray(), tgt: ti.types.ndarray()):
    for i in range(TRAIN_BATCH):
        r = int(ti.random() * MAX_SPLATS)
        emb = pe(s_pos[r])
        for k in range(24):
            inp[i, k] = emb[k]
        val = s_rad[r]
        tgt[i, 0] = val.x
        tgt[i, 1] = val.y
        tgt[i, 2] = val.z


# ------------------------------------------------------------
# NEURAL INFERENCE (LOW RES)
# ------------------------------------------------------------
@ti.kernel
def render_neural():
    for x, y in neural_out:
        fx = int(x * (RES_X / INTERNAL_X))
        fy = int(y * (RES_Y / INTERNAL_Y))
        if mat_id[fx, fy] != -1:
            p = pos[fx, fy]
            emb = pe(p)

            # MLP
            h1 = ti.Vector([0.0] * 32)
            for i in range(32):
                v = b1[i]
                for j in range(24):
                    v += W1[i, j] * emb[j]
                h1[i] = max(0.0, v)

            h2 = ti.Vector([0.0] * 32)
            for i in range(32):
                v = b2[i]
                for j in range(32):
                    v += W2[i, j] * h1[j]
                h2[i] = max(0.0, v)

            out = ti.Vector([0.0] * 3)
            for i in range(3):
                v = b3[i]
                for j in range(32):
                    v += W3[i, j] * h2[j]
                out[i] = max(0.0, v)

            neural_out[x, y] = out
        else:
            neural_out[x, y] = [0, 0, 0]


# ------------------------------------------------------------
# UPSAMPLE + TAA
# ------------------------------------------------------------
@ti.kernel
def upsample_and_accumulate():
    for x, y in display_image:
        # Integer-safe mapping
        fx = int(x * INTERNAL_X // RES_X)
        fy = int(y * INTERNAL_Y // RES_Y)

        fx = min(max(fx, 0), INTERNAL_X - 1)
        fy = min(max(fy, 0), INTERNAL_Y - 1)

        cur = neural_out[fx, fy]

        old = prev_image[x, y]
        alpha = 0.1
        if frame_idx[None] < 2:
            alpha = 1.0

        blended = old * (1 - alpha) + cur * alpha
        prev_image[x, y] = blended

        # Simple tone map (cleaner for debugging)
        display_image[x, y] = blended / (blended + 1.0)


# ------------------------------------------------------------
# MAIN
# ------------------------------------------------------------
def main():
    model = NeuralCore().cuda()
    opt = optim.Adam(model.parameters(), lr=0.005)

    in_t = torch.zeros((TRAIN_BATCH, 24), device="cuda")
    tg_t = torch.zeros((TRAIN_BATCH, 3), device="cuda")

    cam_pos[None] = [0, 0, 3.5]
    cam_look[None] = [0, 0, 0]
    light_pos[None] = [0, 1.8, 0]
    box_pos[None] = [0, -0.6, 0]

    gui = ti.GUI("Neural Cornell", res=(RES_X, RES_Y))
    f = 0

    while gui.running:
        frame_idx[None] = f

        render_gbuffer()
        render_training_rays()

        # train
        fill_train_data(in_t, tg_t)
        opt.zero_grad()
        pred = model(in_t)
        loss = nn.MSELoss()(pred, tg_t)
        loss.backward()
        opt.step()

        # sync weights to Taichi
        W1.from_numpy(model.net[0].weight.cpu().detach().numpy())
        b1.from_numpy(model.net[0].bias.cpu().detach().numpy())
        W2.from_numpy(model.net[3].weight.cpu().detach().numpy())
        b2.from_numpy(model.net[3].bias.cpu().detach().numpy())
        W3.from_numpy(model.net[6].weight.cpu().detach().numpy())
        b3.from_numpy(model.net[6].bias.cpu().detach().numpy())

        render_neural()
        upsample_and_accumulate()

        gui.set_image(display_image)
        gui.show()
        f += 1


if __name__ == "__main__":
    main()
