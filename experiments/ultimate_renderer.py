import math

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# --- CONFIG ---
RES_X, RES_Y = 320, 240
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- HYBRID SETTINGS ---
CASCADE_DIST = 0.6  # Physics Radius
LOOKAHEAD = 10.0  # Oracle Frames

print(f"Running FIXED Ultimate Renderer on: {DEVICE}")


# --- 1. MODEL (Restored Brain Power) ---
class HashEmbedder(nn.Module):
    def __init__(self, num_levels=12, base_res=16, max_res=512, log2_hashmap_size=14):
        # Increased size back to 14 to stop the "Green Blur"
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
        # Normalize P to [0,1]
        return self.net(
            torch.cat([self.embedder(torch.clamp((p + 2) / 4, 0, 1)), n], dim=-1)
        )


# --- 2. SCENE (Cornell Box) ---
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


# --- 3. CAMERA ---
class Camera:
    def __init__(self):
        self.p = torch.tensor([0.0, 0.0, 3.5]).to(DEVICE)
        self.y = 0.0
        self.v = torch.tensor([0.0, 0.0, 0.0]).to(DEVICE)
        self.av = 0.0
        self.lp = self.p.clone()
        self.ly = 0.0

    def update(self, k):
        if k.get(ord("q")):
            self.y -= 0.05
        if k.get(ord("e")):
            self.y += 0.05
        c, s = math.cos(self.y), math.sin(self.y)
        f, r = torch.tensor([s, 0, -c]).to(DEVICE), torch.tensor([c, 0, s]).to(DEVICE)
        if k.get(ord("w")):
            self.p += f * 0.1
        if k.get(ord("s")):
            self.p -= f * 0.1
        if k.get(ord("a")):
            self.p -= r * 0.1
        if k.get(ord("d")):
            self.p += r * 0.1
        self.v = self.p - self.lp
        self.av = self.y - self.ly
        self.lp = self.p.clone()
        self.ly = self.y

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


# --- 4. INTEGRATOR (Albedo Demodulation Fix) ---
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

    # --- CASCADE BOUNCE ---
    r = torch.randn_like(n)
    r = r / r.norm(dim=1, keepdim=True)
    r[((r * n).sum(1, keepdim=True) < 0).squeeze()] *= -1
    tb, nb, cb, mb = sc.intersect(p + n * 1e-2, r)

    is_near = mb & (tb < CASCADE_DIST)
    pb = p + r * tb.unsqueeze(1)

    # --- DEMODULATION FIX ---
    # The Network predicts IRRADIANCE (White Light), not RADIANCE.
    # We multiply by Albedo (cb) OUTSIDE the network.
    # This prevents the network from baking in the wall color.

    ai_irradiance = md(pb, nb)
    indirect = ai_irradiance * cb  # <-- Apply Paint Here

    if is_near.any():
        ldb = sc.light - pb
        db = ldb.norm(dim=1, keepdim=True)
        ldb = ldb / db
        tsb, _, _, smb = sc.intersect(pb + nb * 1e-2, ldb)

        # Physics Calculation
        db_light = (
            (8.0 / (db**2 + 1))
            * torch.clamp((nb * ldb).sum(1, keepdim=True), 0, 1)
            * cb
        )
        db_light[smb & (tsb < db.squeeze(1))] = 0

        # Recurse: (Physics + AI) * Albedo
        # Note: We mix Physics Direct + AI Indirect, then apply Albedo
        combined_irr = db_light + (md(pb, nb) * cb)
        indirect[is_near] = combined_irr[is_near]

    return (
        p,
        n,
        c,
        direct,
        (direct + indirect) * c,
        m,
    )  # Apply Albedo 'c' for final bounce


# --- 5. RENDER LOOP ---
def render(sc, cam, md, opt):
    # Standard Render
    ro, rd = cam.rays()
    p, n, c, d, tgt, v = trace_hybrid(sc, md, ro, rd)

    # Loss: We train against the DEMODULATED target
    # We want the network to predict (Target / Albedo), not Target.
    # This forces it to learn "Light Only".

    # Avoid div by zero
    safe_c = c + 1e-4
    training_target = tgt / safe_c

    l_curr = nn.MSELoss()(md(p, n)[v], training_target[v])

    # Oracle Logic
    is_act = (cam.v.norm() > 1e-3) | (abs(cam.av) > 1e-3)
    orc_img = np.zeros((RES_Y, RES_X, 3))

    if is_act:
        fp = cam.p + cam.v * LOOKAHEAD
        fy = cam.y + cam.av * LOOKAHEAD
        fo, fd = cam.rays(fp, fy)
        idx = torch.randperm(fo.shape[0])[: int(fo.shape[0] * 0.25)]
        fp_s, fn_s, fc_s, _, ft_s, fv_s = trace_hybrid(sc, md, fo[idx], fd[idx])

        # Train Oracle on Demodulated Target too
        safe_fc = fc_s + 1e-4
        ft_target = ft_s / safe_fc

        l_fut = nn.MSELoss()(md(fp_s, fn_s)[fv_s], ft_target[fv_s])
        loss = l_curr + l_fut * 5.0

        # Debug View
        with torch.no_grad():
            _, _, _, d_f, t_f, _ = trace_hybrid(sc, md, fo, fd)
            # Viz shows the raw light + albedo
            orc_img = (
                (d_f + (md(fo + fd * 2, torch.zeros_like(fo)) * 0.8))
                .reshape(RES_Y, RES_X, 3)
                .cpu()
                .numpy()
            )
    else:
        loss = l_curr

    opt.zero_grad()
    loss.backward()
    opt.step()

    fin = d + (md(p, n).detach() * c)  # Apply albedo for display
    img = (fin / (fin + 1)).pow(1 / 2.2).reshape(RES_Y, RES_X, 3).cpu().numpy()
    return img, orc_img, loss.item(), is_act


# --- 6. MAIN ---
def main():
    sc = Scene()
    cam = Camera()
    md = HashNRC().to(DEVICE)
    opt = optim.Adam(md.parameters(), lr=1e-2)
    print("--- FIXED HYBRID RENDERER ---")
    while True:
        k = {}
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break
        for c in [ord("w"), ord("a"), ord("s"), ord("d"), ord("q"), ord("e")]:
            if key == c:
                k[c] = True

        cam.update(k)
        img, orc, l, act = render(sc, cam, md, opt)

        ui = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        th, tw = RES_Y // 3, RES_X // 3
        thm = cv2.resize(
            cv2.cvtColor(orc.astype(np.float32), cv2.COLOR_RGB2BGR), (tw, th)
        )
        cv2.rectangle(
            thm, (0, 0), (tw - 1, th - 1), (0, 255, 0) if act else (0, 0, 255), 1
        )
        ui[-th - 10 : -10, -tw - 10 : -10] = thm
        cv2.putText(ui, f"Albedo-Aware NRC", (10, 20), 0, 0.5, (0, 255, 0), 1)
        cv2.imshow("Final Renderer", ui)


if __name__ == "__main__":
    main()
