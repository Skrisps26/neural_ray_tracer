import torch
import torch.nn as nn
import numpy as np
from config import RES_X, RES_Y

def trace_and_shade(scene, model, ray_o, ray_d):
    # 1. Primary Hit
    t, normal, albedo, mask = scene.intersect(ray_o, ray_d)
    pos = ray_o + ray_d * t.unsqueeze(1)

    # 2. Direct Light + Shadows
    L_dir = scene.light_pos - pos
    L_dist = torch.norm(L_dir, dim=1, keepdim=True)
    L_dir = L_dir / L_dist

    ndotl = torch.clamp(torch.sum(normal * L_dir, dim=1, keepdim=True), 0.0, 1.0)

    # Shadow Ray
    shadow_o = pos + normal * 0.01
    t_shadow, _, _, shadow_mask = scene.intersect(shadow_o, L_dir)

    in_shadow = shadow_mask & (t_shadow < L_dist.squeeze(1))
    direct = (scene.light_intensity / (L_dist**2 + 1.0)) * ndotl * albedo

    direct[in_shadow] = 0.0
    direct[~mask] = 0.0

    # 3. Bounce Ray (Bootstrap)
    rand = torch.randn_like(normal)
    rand = rand / torch.norm(rand, dim=1, keepdim=True)
    if_facing = torch.sum(rand * normal, dim=1, keepdim=True) < 0
    rand[if_facing.squeeze()] *= -1

    t_bounce, norm_B, col_B, mask_B = scene.intersect(pos + normal * 0.01, rand)
    pos_B = pos + rand * t_bounce.unsqueeze(1)

    with torch.no_grad():
        L_dir_B = scene.light_pos - pos_B
        dist_B = torch.norm(L_dir_B, dim=1, keepdim=True)
        L_dir_B_norm = L_dir_B / dist_B

        direct_B = (
            (scene.light_intensity / (dist_B**2 + 1.0))
            * torch.clamp(torch.sum(norm_B * L_dir_B_norm, 1, keepdim=True), 0, 1)
            * col_B
        )

        t_s_B, _, _, s_m_B = scene.intersect(pos_B + norm_B * 0.01, L_dir_B_norm)
        is_shadow_B = s_m_B & (t_s_B < dist_B.squeeze(1))
        direct_B[is_shadow_B] = 0.0

        indirect_B = model(pos_B, norm_B)
        target = (direct_B + indirect_B) * albedo

    return pos, normal, albedo, direct, target, mask & mask_B


def render_loop(scene, camera, model, optimizer):
    # --- A. RENDER CURRENT VIEW ---
    ray_o, ray_d = camera.get_rays()
    pos, normal, albedo, direct, target, valid = trace_and_shade(
        scene, model, ray_o, ray_d
    )
    pred_indirect = model(pos, normal)

    loss_current = nn.MSELoss()(pred_indirect[valid], target[valid])

    # --- B. ORACLE (FUTURE VIEW) ---
    is_moving = torch.norm(camera.velocity) > 0.001
    is_turning = abs(camera.angular_velocity) > 0.001

    oracle_img = np.zeros((RES_Y, RES_X, 3), dtype=np.float32)

    if is_moving or is_turning:
        future_yaw = camera.yaw + (camera.angular_velocity * 10.0)
        future_pos = camera.pos + (camera.velocity * 10.0)

        f_o, f_d = camera.get_rays(pos_override=future_pos, yaw_override=future_yaw)
        f_pos, f_norm, f_albedo, f_direct, f_target, f_valid = trace_and_shade(
            scene, model, f_o, f_d
        )
        f_pred = model(f_pos, f_norm)

        loss_future = nn.MSELoss()(f_pred[f_valid], f_target[f_valid])
        total_loss = loss_current + (loss_future * 5.0)

        f_final = f_direct + (f_pred * f_albedo)
        oracle_img = f_final.reshape(RES_Y, RES_X, 3).detach().cpu().numpy()
    else:
        total_loss = loss_current

    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    final = direct + (pred_indirect * albedo)
    img = final.reshape(RES_Y, RES_X, 3).detach().cpu().numpy()

    def tonemap(im):
        im = im / (im + 1.0)
        return np.power(im, 1.0 / 2.2)

    return tonemap(img), tonemap(oracle_img), total_loss.item()
