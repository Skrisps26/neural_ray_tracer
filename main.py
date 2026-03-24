import math
import cv2
import numpy as np
import torch
import torch.optim as optim
from config import RES_X, RES_Y, DEVICE

class Camera:
    def __init__(self):
        self.pos = torch.tensor([0.0, 0.0, 3.5], device=DEVICE)
        self.yaw = 0.0
        self.velocity = torch.tensor([0.0, 0.0, 0.0], device=DEVICE)
        self.angular_velocity = 0.0
        self.last_pos = self.pos.clone()
        self.last_yaw = 0.0

    def update(self, keys):
        rot_speed = 0.05
        if keys.get(ord("q")): self.yaw -= rot_speed
        if keys.get(ord("e")): self.yaw += rot_speed

        c, s = math.cos(self.yaw), math.sin(self.yaw)
        forward = torch.tensor([s, 0, -c], device=DEVICE)
        right = torch.tensor([c, 0, s], device=DEVICE)

        speed = 0.1
        move_dir = torch.tensor([0.0, 0.0, 0.0], device=DEVICE)
        if keys.get(ord("w")): move_dir += forward
        if keys.get(ord("s")): move_dir -= forward
        if keys.get(ord("a")): move_dir -= right
        if keys.get(ord("d")): move_dir += right

        self.pos += move_dir * speed
        self.velocity = self.pos - self.last_pos
        self.angular_velocity = self.yaw - self.last_yaw
        self.last_pos = self.pos.clone()
        self.last_yaw = self.yaw

    def get_rays(self, pos_override=None, yaw_override=None):
        origin = self.pos if pos_override is None else pos_override
        yaw = self.yaw if yaw_override is None else yaw_override

        y, x = torch.meshgrid(
            torch.linspace(1, -1, RES_Y, device=DEVICE),
            torch.linspace(-1, 1, RES_X, device=DEVICE),
            indexing="ij",
        )

        rx, ry, rz = x, y, -torch.ones_like(x)
        angle = torch.tensor(yaw, device=DEVICE)
        c, s = torch.cos(angle), torch.sin(angle)

        rot_x = rx * c - rz * s
        rot_z = rx * s + rz * c

        ray_d = torch.stack([rot_x, ry, rot_z], dim=-1)
        ray_d = ray_d / torch.norm(ray_d, dim=-1, keepdim=True)
        ray_o = origin.expand_as(ray_d).reshape(-1, 3)
        ray_d = ray_d.reshape(-1, 3)
        return ray_o, ray_d

def main():
    from model import HashNRC
    from scene import Scene
    from renderer import render_loop

    scene = Scene()
    camera = Camera()
    model = HashNRC().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    print("--- DEBUG MODE ---")
    print("Small Window = What the AI is predicting (The Future)")

    while True:
        keys = {}
        key = cv2.waitKey(1) & 0xFF
        for k in [ord("w"), ord("s"), ord("a"), ord("d"), ord("q"), ord("e")]:
            if key == k: keys[k] = True
        if key == 27: break

        camera.update(keys)
        img, oracle_view, loss = render_loop(scene, camera, model, optimizer)

        ui = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        thumb_h, thumb_w = RES_Y // 3, RES_X // 3
        thumb = cv2.resize(cv2.cvtColor(oracle_view, cv2.COLOR_RGB2BGR), (thumb_w, thumb_h))
        cv2.rectangle(thumb, (0, 0), (thumb_w - 1, thumb_h - 1), (0, 255, 0), 1)

        y_offset, x_offset = RES_Y - thumb_h - 10, RES_X - thumb_w - 10
        ui[y_offset : y_offset + thumb_h, x_offset : x_offset + thumb_w] = thumb

        active = (torch.norm(camera.velocity).item() > 0.001) or (abs(camera.angular_velocity) > 0.001)
        cv2.putText(ui, f"Oracle: {'ACTIVE' if active else 'SLEEP'}", (x_offset, y_offset - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        cv2.imshow("Debug View", ui)

if __name__ == "__main__":
    main()
