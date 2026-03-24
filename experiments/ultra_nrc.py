import math

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# --- CONFIG ---
RES_X, RES_Y = 320, 240
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CASCADE_DIST = 0.6
LOOKAHEAD = 10.0
TRAIN_BATCH = 2048  # Smaller batch, but smarter training

print(f"Running ULTIMATE MANIFOLD NRC on: {DEVICE}")


# --- 1. MODEL (Standard Hash Grid) ---
class HashEmbedder(nn.Module):
    def __init__(self, num_levels=12, base_res=16, max_res=1024, log2_hashmap_size=16):
        super().__init__()
        self.resolutions = [
            int(
                base_res
                * (np.exp((np.log(max_res) - np.log(base_res)) / (num_levels - 1))) ** i
            )
            for i in range(num_levels)
        ]
        self.embeddings = nn.ParameterList(
            [
                nn.Parameter(
                    torch.FloatTensor(2**log2_hashmap_size, 2).uniform_(-1e-4, 1e-4)
                )
                for _ in range(num_levels)
            ]
        )
        self.hashmap_size = 2**log2_hashmap_size

    def forward(self, x):
        outputs = []
        for i, res in enumerate(self.resolutions):
            scaled = x * res
            x0 = torch.floor(scaled).long()
            h = (
                (x0 * torch.tensor([1, 2654435761, 805459861], device=x.device))[:, 0]
                ^ (x0 * torch.tensor([1, 2654435761, 805459861], device=x.device))[:, 1]
                ^ (x0 * torch.tensor([1, 2654435761, 805459861], device=x.device))[:, 2]
            ) % self.hashmap_size
            outputs.append(self.embeddings[i][h])
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

    def forward(self, p, n):
        return self.net(
            torch.cat([self.embedder(torch.clamp((p + 2) / 4, 0, 1)), n], dim=-1)
        )


# --- 2. SCENE ---
class Scene:
    def __init__(self):
        self.planes = (
            torch.tensor(
                [[1, 0, 0, 2], [-1, 0, 0, 2], [0, 1, 0, 2], [0, -1, 0, 2], [0, 0, 1, 2]]
            )
            .float()
            .to(DEVICE)
        )
        self.colors = (
            torch.tensor(
                [
                    [0.8, 0.1, 0.1],
                    [0.1, 0.8, 0.1],
                    [0.8, 0.8, 0.8],
                    [0.8, 0.8, 0.8],
                    [0.8, 0.8, 0.8],
                ]
            )
            .float()
            .to(DEVICE)
        )
        self.light = torch.tensor([0.0, 1.9, -1.0]).float().to(DEVICE)
        self.boxes = [
            {
                "s": [0.6, 1.2, 0.6],
                "p": [-0.6, -1.4, -0.6],
                "r": 20,
                "c": [0.8, 0.8, 0.8],
            },
            {
                "s": [0.6, 0.6, 0.6],
                "p": [0.6, -1.7, 0.3],
                "r": -20,
                "c": [0.8, 0.8, 0.8],
            },
        ]
        for b in self.boxes:
            th = math.radians(b["r"])
            c, s = math.cos(th), math.sin(th)
            b["R"] = torch.tensor([[c, 0, s], [0, 1, 0], [-s, 0, c]]).float().to(DEVICE)
            b["iR"] = b["R"].T

    def intersect(self, ro, rd):
        den = torch.where(
            rd @ self.planes[:, :3].T == 0,
            torch.tensor(1e-5).to(DEVICE),
            rd @ self.planes[:, :3].T,
        )
        t = -(ro @ self.planes[:, :3].T + self.planes[:, 3]) / den
        t_rm, id_rm = torch.min(
            torch.where(t > 1e-3, t, torch.tensor(100.0).to(DEVICE)), 1
        )
        tm, nm, cm, mm = t_rm, self.planes[id_rm, :3], self.colors[id_rm], t_rm < 99
        for b in self.boxes:
            ol, dl = (ro - torch.tensor(b["p"]).to(DEVICE)) @ b["R"], rd @ b["R"]
            inv = 1 / (dl + 1e-6)
            h = torch.tensor(b["s"]).to(DEVICE) / 2
            t0, t1 = (-h - ol) * inv, (h - ol) * inv
            tn = torch.max(torch.min(t0, t1), 1)[0]
            tf = torch.min(torch.max(t0, t1), 1)[0]
            hit = (tf > tn) & (tn > 1e-3)
            cl = hit & (tn < tm)
            tm = torch.where(cl, tn, tm)
            mm = mm | hit
            nm = torch.where(
                cl.unsqueeze(1),
                (
                    torch.where(
                        torch.abs(ol + dl * tn.unsqueeze(1)) - h > -1e-3,
                        torch.sign(ol + dl * tn.unsqueeze(1)),
                        torch.zeros_like(ol),
                    )
                )
                @ b["iR"],
                nm,
            )
            cm = torch.where(cl.unsqueeze(1), torch.tensor(b["c"]).to(DEVICE), cm)
        return tm, nm, cm, mm


class Camera:
    def __init__(self):
        self.p = torch.tensor([0.0, 0.0, 3.5]).to(DEVICE)
        self.y = 0.0
        self.v = torch.tensor([0.0, 0.0, 0.0]).to(DEVICE)
        self.av = 0.0

    def update(self, k):
        if k.get(ord("q")):
            self.y -= 0.05
        if k.get(ord("e")):
            self.y += 0.05
        c, s = math.cos(self.y), math.sin(self.y)
        f, r = torch.tensor([s, 0, -c]).to(DEVICE), torch.tensor([c, 0, s]).to(DEVICE)
        self.p += (f if k.get(ord("w")) else 0) * 0.1
        self.p -= (f if k.get(ord("s")) else 0) * 0.1
        self.p -= (r if k.get(ord("a")) else 0) * 0.1
        self.p += (r if k.get(ord("d")) else 0) * 0.1
        self.v = torch.tensor([0.0, 0.0, 0.0]).to(DEVICE)  # Simplified for stability

    def rays(self, po=None, yo=None):
        po = self.p if po is None else po
        yo = self.y if yo is None else yo
        y, x = torch.meshgrid(
            torch.linspace(1, -1, RES_Y).to(DEVICE),
            torch.linspace(-1, 1, RES_X).to(DEVICE),
            indexing="ij",
        )
        rx, ry, rz = x, y, -torch.ones_like(x)
        c, s = math.cos(yo), math.sin(yo)
        d = torch.stack([rx * c - rz * s, ry, rx * s + rz * c], -1)
        return po.expand_as(d).reshape(-1, 3), (
            d / d.norm(dim=-1, keepdim=True)
        ).reshape(-1, 3)


# --- 3. CORE LOGIC ---
def trace_hybrid(sc, md, ro, rd):
    t, n, c, m = sc.intersect(ro, rd)
    p = ro + rd * t.unsqueeze(1)

    # Direct Light
    ld = sc.light - p
    dist = ld.norm(dim=1, keepdim=True)
    ld = ld / dist
    ts, _, _, sm = sc.intersect(p + n * 1e-2, ld)
    direct = (
        (8.0 / (dist**2 + 1)) * torch.clamp((n * ld).sum(1, keepdim=True), 0, 1) * c
    )
    direct[sm & (ts < dist.squeeze(1))] = 0
    direct[~m] = 0

    # Indirect (Cascade)
    r = torch.randn_like(n)
    r = r / r.norm(dim=1, keepdim=True)
    r[((r * n).sum(1, keepdim=True) < 0).squeeze()] *= -1
    tb, nb, cb, mb = sc.intersect(p + n * 1e-2, r)
    is_near = mb & (tb < CASCADE_DIST)

    pb = p + r * tb.unsqueeze(1)
    ai_irr = md(pb, nb)
    indirect = ai_irr * cb

    if is_near.any():
        ldb = sc.light - pb
        db = ldb.norm(dim=1, keepdim=True)
        ldb = ldb / db
        tsb, _, _, smb = sc.intersect(pb + nb * 1e-2, ldb)
        db_light = (
            (8.0 / (db**2 + 1))
            * torch.clamp((nb * ldb).sum(1, keepdim=True), 0, 1)
            * cb
        )
        db_light[smb & (tsb < db.squeeze(1))] = 0
        combined = db_light + (md(pb, nb) * cb)
        indirect[is_near] = combined[is_near]

    return p, n, c, direct, (direct + indirect) * c, m


# --- 4. MANIFOLD TRAINING LOOP ---
def render(sc, cam, md, opt):
    # 1. SAMPLE PIXELS
    y_idx = torch.randint(0, RES_Y, (TRAIN_BATCH,)).to(DEVICE)
    x_idx = torch.randint(0, RES_X, (TRAIN_BATCH,)).to(DEVICE)
    y_coord = torch.linspace(1, -1, RES_Y).to(DEVICE)[y_idx]
    x_coord = torch.linspace(-1, 1, RES_X).to(DEVICE)[x_idx]
    rx, ry, rz = x_coord, y_coord, -torch.ones_like(x_coord)

    angle = torch.tensor(cam.y, device=DEVICE)
    cos, sin = torch.cos(angle), torch.sin(angle)
    rd = torch.stack([rx * cos - rz * sin, ry, rx * sin + rz * cos], -1)
    rd = rd / rd.norm(dim=-1, keepdim=True)
    ro = cam.p.expand_as(rd)

    # 2. GENERATE NEIGHBORS (The "Consistency" Trick)
    # For every pixel we train, we create a "Ghost Neighbor" 1cm away.
    # This forces the network to learn smooth gradients, not checkerboards.
    ro_neighbor = ro + (torch.randn_like(ro) * 0.01)

    # Trace Main Rays
    p, n, c, d, tgt, v = trace_hybrid(sc, md, ro, rd)

    # Trace Neighbor Rays (Reuse Same Direction)
    p_n, n_n, _, _, _, v_n = trace_hybrid(sc, md, ro_neighbor, rd)

    # 3. CALCULATE LOSSES
    train_tgt = tgt / (c + 1e-4)
    pred_main = md(p, n)
    pred_neighbor = md(p_n, n_n)

    # Loss A: Physics Accuracy (Standard)
    loss_physics = ((pred_main[v] - train_tgt[v]) ** 2).mean()

    # Loss B: Spatial Consistency (The Ultimate Fix)
    # IF the normals are the same (same surface), the lighting MUST be similar.
    # This kills the checkerboard pattern instantly.
    normal_dot = (n[v] * n_n[v]).sum(dim=1)
    is_same_surface = normal_dot > 0.95

    # We penalize the DIFFERENCE between neighbors.
    loss_smoothness = (
        (pred_main[v][is_same_surface] - pred_neighbor[v][is_same_surface]) ** 2
    ).mean()

    # Combine: 80% Physics, 20% Smoothness
    total_loss = loss_physics + (loss_smoothness * 0.2)

    opt.zero_grad()
    total_loss.backward()
    opt.step()

    # 4. INFERENCE
    with torch.no_grad():
        ro_f, rd_f = cam.rays()
        p_f, n_f, c_f, d_f, _, _ = trace_hybrid(sc, md, ro_f, rd_f)
        final = d_f + (md(p_f, n_f) * c_f)

    img = (final / (final + 1)).pow(1 / 2.2).reshape(RES_Y, RES_X, 3).cpu().numpy()
    return img, total_loss.item()


# --- 5. MAIN ---
def main():
    sc = Scene()
    cam = Camera()
    md = HashNRC().to(DEVICE)
    opt = optim.Adam(md.parameters(), lr=1e-2)
    print("--- ULTIMATE MANIFOLD NRC ---")
    print("Features: Spatial Consistency, Zero patches.")

    while True:
        k = {}
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break
        for c in [ord("w"), ord("a"), ord("s"), ord("d"), ord("q"), ord("e")]:
            if key == c:
                k[c] = True

        cam.update(k)
        img, l = render(sc, cam, md, opt)
        cv2.imshow("Ultimate NRC", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))


if __name__ == "__main__":
    main()
