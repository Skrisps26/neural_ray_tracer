import torch
import math
from config import DEVICE

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

        t_near = torch.max(torch.min(t0, t1), dim=1)[0]
        t_far = torch.min(torch.max(t0, t1), dim=1)[0]

        hit = (t_far > t_near) & (t_near > 0.001)

        # 3. Calculate Normal
        hit_p = o_local + d_local * t_near.unsqueeze(1)
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
