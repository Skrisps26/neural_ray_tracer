import math

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# --- CONFIG ---
RES_X, RES_Y = 320, 240
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_LOOKAHEAD = 5.0
TRAIN_BATCH = 8192  # Heavy training for zero noise


# --- 1. THE BRAIN (Radiance Cache) ---
class NeuralRadiance(nn.Module):
    def __init__(self):
        super().__init__()
        # 65k Hash Map to kill checkerboard patterns
        self.embedder = nn.Embedding(65536, 16)
        self.net = nn.Sequential(
            nn.Linear(16 + 3, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 3),
            nn.Sigmoid(),
        )

    def forward(self, pos, normal):
        pos, normal = pos.float(), normal.float()
        scaled = (pos * 12.0).long()
        idx = (
            scaled[:, 0] * 73856093 ^ scaled[:, 1] * 19349663 ^ scaled[:, 2] * 83492791
        ) % 65536
        feat = self.embedder(idx)
        return self.net(torch.cat([feat, normal], dim=-1))


# --- 2. PREDICTIVE CAMERA ---
class Camera:
    def __init__(self):
        self.pos = torch.tensor([0.0, 0.0, 3.5], device=DEVICE)
        self.yaw = 0.0
        self.vel = torch.tensor([0.0, 0.0, 0.0], device=DEVICE)
        self.avel = 0.0
        self.lp = self.pos.clone()
        self.ly = self.yaw

    def update(self, keys):
        self.lp = self.pos.clone()
        self.ly = self.yaw
        c, s = math.cos(self.yaw), math.sin(self.yaw)
        f, r = (
            torch.tensor([s, 0, -c], device=DEVICE),
            torch.tensor([c, 0, s], device=DEVICE),
        )

        move = torch.tensor([0.0, 0.0, 0.0], device=DEVICE)
        if keys.get(ord("w")):
            move += f * 0.1
        if keys.get(ord("s")):
            move -= f * 0.1
        if keys.get(ord("a")):
            move -= r * 0.1
        if keys.get(ord("d")):
            move += r * 0.1
        self.pos += move
        if keys.get(ord("q")):
            self.yaw -= 0.05
        if keys.get(ord("e")):
            self.yaw += 0.05

        self.vel = self.pos - self.lp
        self.avel = self.yaw - self.ly

    def get_future_state(self, amount):
        return self.pos + (self.vel * amount), self.yaw + (self.avel * amount)

    def get_rays(self, po=None, yo=None):
        po = self.pos if po is None else po
        yo = self.yaw if yo is None else yo
        y, x = torch.meshgrid(
            torch.linspace(1, -1, RES_Y, device=DEVICE),
            torch.linspace(-1, 1, RES_X, device=DEVICE),
            indexing="ij",
        )
        rx, ry, rz = x, y, -torch.ones_like(x)
        c, s = math.cos(yo), math.sin(yo)
        rd = torch.stack([rx * c - rz * s, ry, rx * s + rz * c], -1)
        rd = rd / (rd.norm(dim=-1, keepdim=True) + 1e-6)
        return po.expand_as(rd).reshape(-1, 3), rd.reshape(-1, 3)


# --- 3. SCENE (Analytical Oracle) ---
class Scene:
    def __init__(self):
        # Cornell Box Walls + Box Logic
        self.planes = torch.tensor(
            [[1, 0, 0, 2], [-1, 0, 0, 2], [0, 1, 0, 2], [0, -1, 0, 2], [0, 0, 1, 2]],
            device=DEVICE,
        ).float()
        self.colors = torch.tensor(
            [
                [0.8, 0.1, 0.1],
                [0.1, 0.8, 0.1],
                [0.8, 0.8, 0.8],
                [0.8, 0.8, 0.8],
                [0.8, 0.8, 0.8],
            ],
            device=DEVICE,
        ).float()
        self.light_pos = torch.tensor([0.0, 1.8, -1.0], device=DEVICE)

    def intersect(self, ro, rd):
        den = rd @ self.planes[:, :3].T
        den = torch.where(den.abs() < 1e-5, torch.sign(den) * 1e-5, den)
        t = -(ro @ self.planes[:, :3].T + self.planes[:, 3]) / den
        tm, idx = torch.min(
            torch.where(t > 0.01, t, torch.tensor(100.0, device=DEVICE)), 1
        )
        return tm, self.planes[idx, :3], self.colors[idx], tm < 99


# --- 4. ENGINE ---
def render(scene, cam, model, opt, frame):
    # --- TRAINING (Future Oracle GI) ---
    f_pos, f_yaw = cam.get_future_state(MAX_LOOKAHEAD)
    fro, frd = cam.get_rays(f_pos, f_yaw)
    sub = torch.randint(0, RES_X * RES_Y, (TRAIN_BATCH,), device=DEVICE)
    fro_s, frd_s = fro[sub], frd[sub]

    p_t, n_t, c_t, m_t = scene.intersect(fro_s, frd_s)
    hit_p = fro_s + frd_s * p_t.unsqueeze(1)

    if m_t.any():
        pred = model(hit_p, n_t)
        with torch.no_grad():
            lp = scene.light_pos - hit_p
            dist = lp.norm(dim=1, keepdim=True)
            lp_n = lp / (dist + 1e-5)
            # ORACLE HARD SHADOWS
            ts, _, _, sm = scene.intersect(hit_p + n_t * 0.01, lp_n)
            vis = torch.ones_like(dist)
            vis[sm & (ts < dist.squeeze(1))] = 0.0
            direct = (
                (60.0 / (dist**2 + 1))
                * torch.clamp((n_t * lp_n).sum(1, True), 0, 1)
                * vis
                * c_t
            )
            target = direct
        loss = nn.MSELoss()(pred, target)
        opt.zero_grad()
        loss.backward()
        opt.step()

    # --- INFERENCE ---
    with torch.no_grad():
        ro, rd = cam.get_rays()
        tm, n, c, m = scene.intersect(ro, rd)
        p = ro + rd * tm.unsqueeze(1)
        shading = model(p, n)
        # Tone Map (ACES approx)
        final = c * shading * 5.0
        final = final / (final + 1.0)
        img = final.reshape(RES_Y, RES_X, 3).cpu().numpy()
        img = np.power(img, 1.0 / 2.2)
    return img


def main():
    sc, cam = Scene(), Camera()
    md = NeuralRadiance().to(DEVICE)
    opt = optim.Adam(md.parameters(), lr=0.01)
    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break
        k = {ord(c): (key == ord(c)) for c in "wasdqijkle"}
        cam.update(k)
        img = render(sc, cam, md, opt, 0)
        cv2.imshow("The Oracle NRC GI", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))


if __name__ == "__main__":
    main()
