import os

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"  # <--- MUST BE FIRST

import cv2
import numpy as np
import torch

from model import NeuralRenderer

# --- CONFIG ---
MODEL_PATH = "model_final_finetuned.pth"  # Make sure this matches your save file
FRAME_IDX = 4
DATA_ROOT = "dataset_indoor_safe"
SCENE_SCALE = 150.0  # Match your training scale!
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_exr(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing: {path}")
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def aces_filmic_tone_mapping(x):
    """
    ACES Tone Mapping is the industry standard for 3D rendering.
    It handles bright highlights (Orange -> Yellow clipping) much better
    than standard Gamma correction.
    """
    a = 2.51
    b = 0.03
    c = 2.43
    d = 0.59
    e = 0.14
    return torch.clamp((x * (a * x + b)) / (x * (c * x + d) + e), 0.0, 1.0)


def infer():
    print(f"--- Running Final Inference ---")

    # 1. Load Model
    model = NeuralRenderer(input_dim=81).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    # 2. Find Files (Same as before)
    scene_name = os.listdir(DATA_ROOT)[4]
    scene_path = os.path.join(DATA_ROOT, scene_name)
    files = sorted([f for f in os.listdir(scene_path) if "_beauty.exr" in f])

    path_beauty = os.path.join(scene_path, files[FRAME_IDX])
    path_albedo = path_beauty.replace("_beauty.exr", "_albedo.exr")
    path_normal = path_beauty.replace("_beauty.exr", "_normal.exr")
    path_pos = path_beauty.replace("_beauty.exr", "_pos.exr")
    path_cam = path_beauty.replace("_beauty.exr", "_cam.npy")

    # 3. Load & Process
    beauty = load_exr(path_beauty)
    albedo = load_exr(path_albedo)
    normal = load_exr(path_normal)
    pos = load_exr(path_pos)
    cam = np.load(path_cam)

    H, W, _ = beauty.shape

    # Convert to Tensors
    pos_t = torch.from_numpy(pos).to(DEVICE).float().view(-1, 3)
    albedo_t = torch.from_numpy(albedo).to(DEVICE).float().view(-1, 3)
    normal_t = torch.from_numpy(normal).to(DEVICE).float().view(-1, 3)

    if cam.ndim == 2:
        cam_pos = cam[:3, 3]
    else:
        cam_pos = cam
    cam_pos_t = torch.from_numpy(cam_pos).to(DEVICE).float()

    # Relative Coords & Scaling
    pos_relative = pos_t - cam_pos_t
    dist = torch.norm(pos_relative, dim=1, keepdim=True)
    mask = (dist < (SCENE_SCALE * 1.5)).float()
    pos_final = (pos_relative * mask) / SCENE_SCALE

    # --- FIX 1: Uncap Emitters (Keep this!) ---
    is_pure_black = albedo_t.mean(dim=1, keepdim=True) < 0.001
    albedo_t = torch.where(is_pure_black, torch.ones_like(albedo_t), albedo_t)

    with torch.no_grad():
        # Predict
        log_irradiance = model(albedo_t, normal_t, pos_final)
        pred_irradiance = torch.expm1(log_irradiance)

        # --- FIX 2: REMOVE EXPOSURE MULTIPLIER ---
        # We delete the "* 1.8" line. The lights are bright enough now.

        # Recombine
        pred_radiance = pred_irradiance * albedo_t

        # --- FIX 3: BETTER TONE MAPPING ---
        # Instead of simple gamma, we use ACES (Standard for Games/Movies)
        # It desaturates brights instead of shifting hue.
        pred_image = aces_filmic_tone_mapping(pred_radiance)

        # --- FIX 4: CONTRAST S-CURVE ---
        # This pushes dark grays (the pot) to black.
        # Power > 1.0 darkens shadows.
        pred_image = torch.pow(pred_image, 1.3)

        # Reshape & Save
        pred_img_np = pred_image.view(H, W, 3).cpu().numpy()
        pred_img_np = (pred_img_np * 255).astype(np.uint8)

    # Save
    cv2.imwrite(
        "infer_final_corrected.png", cv2.cvtColor(pred_img_np, cv2.COLOR_RGB2BGR)
    )
    print("Saved 'infer_final_corrected.png'")


if __name__ == "__main__":
    infer()
