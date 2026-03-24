import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# --- CONFIG ---
RES_X, RES_Y = 320, 240
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_LOOKAHEAD = 5.0
TRAIN_BATCH = 4096


# --- 1. THE BRAIN (High-Res Memory) ---
class NeuralRadiance(nn.Module):
    def __init__(self):
        super().__init__()
        # 32k Hash Size to kill the checkerboard pattern
        self.embedder = nn.Embedding(32768, 16)
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
        scaled = (pos * 8.0).long()  # Sharper grid scaling
        idx = (
            scaled[:, 0] * 73856093 ^ scaled[:, 1] * 19349663 ^ scaled[:, 2] * 83492791
        ) % 32768
        feat = self.embedder(idx)
        return self.net(torch.cat([feat, normal], dim=-1))


# --- 2. THE PREDICTIVE CAMERA ---
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
        m = torch.tensor([0.0, 0.0, 0.0], device=DEVICE)
        if keys.get(ord("w")):
            m[2] -= 0.1
        if keys.get(ord("s")):
            m[2] += 0.1
        if keys.get(ord("a")):
            m[0] -= 0.1
        if keys.get(ord("d")):
            m[0] += 0.1
        if keys.get(ord("q")):
            self.yaw -= 0.05
        if keys.get(ord("e")):
            self.yaw += 0.05
        self.pos += m
        self.vel = self.pos - self.lp  #
        self.avel = self.yaw - self.ly

    def get_future_state(self, amount):
        return self.pos + (self.vel * amount), self.yaw + (self.avel * amount)


# --- 3. THE WORLD (Lumen Hybrid) ---
class CornellBox:
    def __init__(self):
        self.boxes = []
        self.colors_list = []
        self.box_vel = []
        # Dense Surface Cache for AI Learning
        self.gauss_pos = torch.zeros(0, 3, device=DEVICE)
        self.gauss_col = torch.zeros(0, 3, device=DEVICE)
        self.gauss_vel = torch.zeros(0, 3, device=DEVICE)

    def add_box(self, center, size, color):
        self.boxes.append(torch.tensor([*center, *size], device=DEVICE))
        self.colors_list.append(torch.tensor(color, device=DEVICE))
        self.box_vel.append(torch.zeros(3, device=DEVICE))
        num_pts = 800  # Higher density surface cache
        pts = (torch.rand(num_pts, 3, device=DEVICE) - 0.5) * torch.tensor(
            size, device=DEVICE
        ) + torch.tensor(center, device=DEVICE)
        self.gauss_pos = torch.cat([self.gauss_pos, pts])
        self.gauss_col = torch.cat(
            [self.gauss_col, torch.tensor(color, device=DEVICE).repeat(num_pts, 1)]
        )
        self.gauss_vel = torch.cat(
            [self.gauss_vel, torch.zeros(num_pts, 3, device=DEVICE)]
        )
        return len(self.boxes) - 1

    def update_physics(self, keys, idx):
        m = torch.tensor([0.0, 0.0, 0.0], device=DEVICE)
        if keys.get(ord("i")):
            m[2] -= 0.1
        if keys.get(ord("k")):
            m[2] += 0.1
        if keys.get(ord("j")):
            m[0] -= 0.1
        if keys.get(ord("l")):
            m[0] += 0.1
        v = m * 0.1
        self.box_vel[idx] = self.box_vel[idx] * 0.9 + v
        self.boxes[idx][:3] += self.box_vel[idx]
        start = idx * 800
        self.gauss_vel[start : start + 800] = self.box_vel[idx]
        self.gauss_pos[start : start + 800] += self.box_vel[idx]


# --- 4. ENGINE ---
def intersect(ro, rd, boxes):
    bt = torch.stack(boxes)
    bn, bx = bt[:, :3] - bt[:, 3:] / 2, bt[:, :3] + bt[:, 3:] / 2
    ird = 1.0 / (rd.unsqueeze(1) + 1e-6)
    t1, t2 = (
        (bn.unsqueeze(0) - ro.unsqueeze(1)) * ird,
        (bx.unsqueeze(0) - ro.unsqueeze(1)) * ird,
    )
    tmin, tmax = (
        torch.max(torch.min(t1, t2), -1)[0],
        torch.min(torch.max(t1, t2), -1)[0],
    )
    hit = (tmax > tmin) & (tmax > 0)
    mt, idx = torch.where(hit, tmin, torch.tensor(float("inf"), device=DEVICE)).min(
        dim=1
    )
    m = mt < float("inf")
    p = ro + rd * mt.unsqueeze(-1)
    lp = p - bt[idx, :3]
    n = torch.where(torch.abs(lp) / (bt[idx, 3:] / 2) >= 0.99, torch.sign(lp), 0.0)
    return p, n / (n.norm(dim=-1, keepdim=True) + 1e-5), idx, m


def render(scene, cam, model, opt, frame, history):
    lookahead = min(MAX_LOOKAHEAD, frame / 10.0)
    colors = torch.stack(scene.colors_list)

    # --- TRAIN (GOD MODE GI) ---
    fg, (fp, fy) = (
        scene.gauss_pos + (scene.gauss_vel * lookahead),
        cam.get_future_state(lookahead),
    )
    y, x = torch.meshgrid(
        torch.linspace(1, -1, RES_Y, device=DEVICE),
        torch.linspace(-1, 1, RES_X, device=DEVICE),
        indexing="ij",
    )
    a = torch.tensor(fy, device=DEVICE)
    c, s = torch.cos(a), torch.sin(a)
    rd = torch.stack([x * c - (-1) * s, y, x * s + (-1) * c], -1)
    rd /= rd.norm(dim=-1, keepdim=True)
    sub = torch.randint(0, RES_X * RES_Y, (TRAIN_BATCH,), device=DEVICE)
    frd, fro = rd.reshape(-1, 3)[sub], fp.expand_as(rd.reshape(-1, 3)[sub])
    p, n, i, m = intersect(fro, frd, scene.boxes)

    if m.any():
        pred = model(p[m], n[m])
        with torch.no_grad():
            # High Contrast Light
            lp = torch.tensor([0.0, 1.8, 0.0], device=DEVICE)
            ld = lp - p[m]
            dist = ld.norm(dim=-1, keepdim=True)
            dir_l = torch.clamp((n[m] * (ld / dist)).sum(-1, True), 0.1, 1.0) * (
                20.0 / (dist**2 + 0.5)
            )
            # Bounce Sample
            b_dir = torch.randn_like(n[m])
            b_dir *= torch.sign((b_dir * n[m]).sum(-1, True))
            g_dist = torch.cdist(p[m] + b_dir * 0.2, fg)
            ind = scene.gauss_col[g_dist.min(1)[1]] * 0.4
            tgt = (colors[i[m]] * dir_l) + ind
        loss = nn.MSELoss()(pred, tgt)
        opt.zero_grad()
        loss.backward()
        opt.step()

    # --- INFERENCE ---
    with torch.no_grad():
        ac = torch.tensor(cam.yaw, device=DEVICE)
        cc, ss = torch.cos(ac), torch.sin(ac)
        rd_c = torch.stack([x * cc - (-1) * ss, y, x * ss + (-1) * cc], -1)
        rd_c /= rd_c.norm(dim=-1, keepdim=True)
        ard, aro = rd_c.reshape(-1, 3), cam.pos.expand_as(rd_c.reshape(-1, 3))
        out = torch.zeros_like(ard)

        # Exact Geometry Resolve
        p_i, n_i, ii, mi = intersect(aro, ard, scene.boxes)
        if mi.any():
            # Neural Light Application
            out[mi] = colors[ii[mi]] * (model(p_i[mi], n_i[mi]) + 0.05)

        curr = out.reshape(RES_Y, RES_X, 3).cpu().numpy()
        # MOTION-SENSITIVE BLUR REMOVAL
        v_speed = (
            cam.vel.norm().item() + abs(cam.avel) + scene.box_vel[-1].norm().item()
        )
        alpha = 0.0 if v_speed > 0.01 else 0.85
        history = curr if history is None else history * alpha + curr * (1.0 - alpha)

        # TONE MAPPING (ACES Approx) to fix blandness
        final = (history * 1.5) / (history * 1.5 + 1.0)
    return final, 0.0


# --- MAIN ---
def main():
    scene, cam = CornellBox(), Camera()
    model, opt = (
        NeuralRadiance().to(DEVICE),
        optim.Adam(NeuralRadiance().to(DEVICE).parameters(), lr=0.01),
    )
    scene.add_box([-1.1, 0, 0], [0.1, 2, 2], [1, 0.1, 0.1])
    scene.add_box([1.1, 0, 0], [0.1, 2, 2], [0.1, 1, 0.1])
    scene.add_box([0, 0, -1.1], [2.2, 2, 0.1], [0.8, 0.8, 0.8])
    scene.add_box([0, -1.1, 0], [2.2, 0.1, 2.2], [0.8, 0.8, 0.8])
    scene.add_box([0, 1.1, 0], [2.2, 0.1, 2.2], [0.8, 0.8, 0.8])
    scene.add_box([0, 1.0, 0], [0.6, 0.05, 0.6], [20, 20, 20])
    p_idx = scene.add_box([0, -0.6, 0], [0.6, 0.6, 0.6], [0.1, 0.4, 0.9])
    frame, hist = 0, None
    while True:
        k = {}
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break
        for c in "wasdqijkle":
            if key == ord(c):
                k[ord(c)] = True
        scene.update_physics(k, p_idx)
        cam.update(k)
        hist, _ = render(scene, cam, model, opt, frame, hist)
        cv2.imshow("Oracle Lumen GI", cv2.cvtColor(hist, cv2.COLOR_RGB2BGR))
        frame += 1


if __name__ == "__main__":
    main()
