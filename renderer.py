import torch
import torch.nn as nn
import numpy as np
from config import RES_X, RES_Y

def gaussian_nll_loss(pred_out, target, valid_mask):
    """
    Computes Gaussian Negative Log Likelihood:
    L = ((pred - target)^2 / (2 * exp(log_var))) + 0.5 * log_var
    """
    if not valid_mask.any():
        return torch.tensor(0.0, device=pred_out.device, requires_grad=True)

    pred_rgb = pred_out[valid_mask, :3]
    log_var = pred_out[valid_mask, 3:]
    target_rgb = target[valid_mask]

    # Stability: Clamp log_var to a reasonable range
    log_var = torch.clamp(log_var, -10.0, 2.0)
    precision = torch.exp(-log_var)

    # NLL components: weighted MSE + log term
    diff_sq = (pred_rgb - target_rgb) ** 2
    loss = 0.5 * (precision * diff_sq + log_var).mean()
    
    return loss

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

    # Use 1D Mask for Indexing
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
        # Target Calculation at B
        L_dir_B = scene.light_pos - pos_B
        dist_B = torch.norm(L_dir_B, dim=1, keepdim=True)
        # Normalize Light Dir B
        L_dir_B_norm = L_dir_B / dist_B

        # Calculate Direct Light at B
        direct_B = (
            (scene.light_intensity / (dist_B**2 + 1.0))
            * torch.clamp(torch.sum(norm_B * L_dir_B_norm, 1, keepdim=True), 0, 1)
            * col_B
        )

        # Shadow Check at B
        t_s_B, _, _, s_m_B = scene.intersect(pos_B + norm_B * 0.01, L_dir_B_norm)

        # Apply 1D Mask Here Too
        is_shadow_B = s_m_B & (t_s_B < dist_B.squeeze(1))
        direct_B[is_shadow_B] = 0.0

        # Indirect at B - we only need the RGB part for the bootstrap target
        out_B = model(pos_B, norm_B)
        indirect_B = out_B[:, :3]
        target = (direct_B + indirect_B) * albedo

    return pos, normal, albedo, direct, target, mask & mask_B


def render_loop(scene, camera, model, optimizer):
    # --- A. RENDER CURRENT VIEW ---
    ray_o, ray_d = camera.get_rays()
    pos, normal, albedo, direct, target, valid = trace_and_shade(
        scene, model, ray_o, ray_d
    )
    pred_out = model(pos, normal)
    pred_indirect = pred_out[:, :3]
    uncertainty = pred_out[:, 3:]

    # Use custom Gaussian NLL loss
    loss_current = gaussian_nll_loss(pred_out, target, valid)

    # --- B. ORACLE (FUTURE VIEW) ---
    is_moving = torch.norm(camera.velocity) > 0.001
    is_turning = abs(camera.angular_velocity) > 0.001

    # We will generate a debug image for the Oracle
    oracle_img = np.zeros((RES_Y, RES_X, 3), dtype=np.float32)

    if is_moving or is_turning:
        # 1. Calculate Future State
        future_yaw = camera.yaw + (
            camera.angular_velocity * 10.0
        )  # Look 10 frames ahead
        future_pos = camera.pos + (camera.velocity * 10.0)

        # 2. Trace Future Rays
        f_o, f_d = camera.get_rays(pos_override=future_pos, yaw_override=future_yaw)

        f_pos, f_norm, f_albedo, f_direct, f_target, f_valid = trace_and_shade(
            scene, model, f_o, f_d
        )
        f_out = model(f_pos, f_norm)
        f_pred_indirect = f_out[:, :3]

        # Use Gaussian NLL loss for the Oracle too
        loss_future = gaussian_nll_loss(f_out, f_target, f_valid)

        # Combine losses
        total_loss = loss_current + (loss_future * 5.0)

        # Generate Oracle Debug Image
        f_final = f_direct + (f_pred_indirect * f_albedo)
        oracle_img = f_final.reshape(RES_Y, RES_X, 3).detach().cpu().numpy()

    else:
        total_loss = loss_current

    # Optimization Step
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    # Tone Map Main Image
    final = direct + (pred_indirect * albedo)
    img = final.reshape(RES_Y, RES_X, 3).detach().cpu().numpy()

    # Common Tone Mapping for both
    def tonemap(im):
        im = im / (im + 1.0)
        return np.power(im, 1.0 / 2.2)

    return tonemap(img), tonemap(oracle_img), total_loss.item()
