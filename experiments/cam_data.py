import os

# --- FIX: ENABLE OPENEXR ---
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

import glob

import cv2
import numpy as np
import torch
from tqdm import tqdm

# --- CONFIG ---
INPUT_ROOT = "dataset_indoor_safe"
OUTPUT_ROOT = "dataset_optimized"
CHUNK_SIZE = 50


def get_projection_matrix(fov, aspect, near=0.1, far=100.0):
    f = 1.0 / np.tan(np.radians(fov) / 2.0)
    proj = np.zeros((4, 4))
    proj[0, 0] = f / aspect
    proj[1, 1] = f
    proj[2, 2] = (far + near) / (near - far)
    proj[2, 3] = (2 * far * near) / (near - far)
    proj[3, 2] = -1.0
    return proj


def compute_flow(pos_world, cam_next_mat, proj_mat, width, height):
    H, W, _ = pos_world.shape
    points_3d = pos_world.reshape(-1, 3)
    ones = np.ones((points_3d.shape[0], 1))
    points_4d = np.hstack([points_3d, ones])

    # World -> View -> Clip
    view_mat = np.linalg.inv(cam_next_mat)
    full_mat = view_mat.T @ proj_mat.T
    clip_space = points_4d @ full_mat

    # Perspective Divide
    w = clip_space[:, 3:4]
    w[np.abs(w) < 1e-5] = 1e-5
    ndc = clip_space[:, :3] / w

    # NDC -> Screen
    screen_x = ((ndc[:, 0] + 1) * 0.5) * width
    screen_y = ((1 - ndc[:, 1]) * 0.5) * height

    # Flow = New Pos - Old Pos
    grid_y, grid_x = np.mgrid[0:H, 0:W]
    flow = np.stack(
        [screen_x - grid_x.flatten(), screen_y - grid_y.flatten()], axis=1
    ).reshape(H, W, 2)

    # Mask invalid (behind camera)
    mask = (w.flatten() < 0).reshape(H, W)
    flow[mask] = 0
    return flow


def load_exr_cv2(path):
    """
    Robustly loads EXR using OpenCV.
    Returns: RGB numpy array (H, W, 3)
    """
    if not os.path.exists(path):
        return None

    try:
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    except Exception:
        return None

    if img is None:
        return None

    # OpenCV loads as BGR. Swap to RGB.
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def save_chunk(data, scene, idx):
    if len(data["beauty"]) == 0:
        return

    # Helper to stack and convert to Float16 Tensor (Images)
    def to_tensor(params):
        arr = np.stack(params)
        return torch.from_numpy(arr).permute(0, 3, 1, 2).half()

    # --- NEW: Helper for simple vectors (N, 3) -> (N, 3, 1, 1) ---
    def to_vec_tensor(params):
        arr = np.stack(params)  # (N, 3)
        # Reshape to (N, 3, 1, 1) so we can subtract it from images easily
        return torch.from_numpy(arr).view(-1, 3, 1, 1).float()

    out_dict = {
        "beauty": to_tensor(data["beauty"]),
        "noisy": to_tensor(data["noisy"]),
        "albedo": to_tensor(data["albedo"]),
        "normal": to_tensor(data["normal"]),
        "pos": to_tensor(data["pos"]),
        "flow": to_tensor(data["flow"]),
        "cam_pos": to_vec_tensor(data["cam_pos"]),  # <--- SAVING CAMERA DATA HERE
    }

    save_path = os.path.join(OUTPUT_ROOT, f"{scene}_chunk_{idx:02d}.pt")
    torch.save(out_dict, save_path)
    # print(f"  Saved {save_path}")


def process_dataset():
    if not os.path.exists(INPUT_ROOT):
        print("Input folder missing.")
        return

    # Identify scenes
    scenes = [
        d for d in os.listdir(INPUT_ROOT) if os.path.isdir(os.path.join(INPUT_ROOT, d))
    ]
    print(f"Found scenes: {scenes}")

    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    PROJ_MAT = get_projection_matrix(fov=60, aspect=1.0)

    for scene in scenes:
        print(f"\nProcessing: {scene}")
        scene_path = os.path.join(INPUT_ROOT, scene)
        files = sorted(glob.glob(os.path.join(scene_path, "*_beauty.exr")))

        chunk_data = {
            "beauty": [],
            "noisy": [],
            "albedo": [],
            "normal": [],
            "pos": [],
            "flow": [],
            "cam_pos": [],  # <--- NEW LIST FOR CAMERA POSITIONS
        }
        chunk_idx = 0
        valid_count = 0

        # Wrap in tqdm for progress bar
        for i in tqdm(range(len(files) - 1)):
            f_beauty = files[i]
            base = f_beauty.replace("_beauty.exr", "")

            # Paths
            paths = {
                "noisy": base + "_noisy.exr",
                "albedo": base + "_albedo.exr",
                "normal": base + "_normal.exr",
                "pos": base + "_pos.exr",
                "cam": base + "_cam.npy",
                "cam_next": files[i + 1].replace("_beauty.exr", "_cam.npy"),
            }

            # 1. Existence Check
            if not all(os.path.exists(p) for p in paths.values()):
                continue

            # 2. Load Data (Using OpenCV)
            beauty = load_exr_cv2(f_beauty)
            noisy = load_exr_cv2(paths["noisy"])
            albedo = load_exr_cv2(paths["albedo"])
            normal = load_exr_cv2(paths["normal"])
            pos = load_exr_cv2(paths["pos"])

            if any(x is None for x in [beauty, noisy, albedo, normal, pos]):
                continue

            try:
                cam_curr = np.load(paths["cam"])
                cam_next = np.load(paths["cam_next"])

                # --- NEW: EXTRACT CAMERA POSITION ---
                # Check if it's a 4x4 matrix or just a position vector
                if cam_curr.ndim == 2 and cam_curr.shape == (4, 4):
                    # Extract translation (top-right column)
                    current_pos = cam_curr[:3, 3]
                else:
                    # Assume it's already a position vector
                    current_pos = cam_curr

                # Append to our list
                chunk_data["cam_pos"].append(current_pos)
                # ------------------------------------

                # 3. Compute Flow
                flow = compute_flow(
                    pos, cam_next, PROJ_MAT, beauty.shape[1], beauty.shape[0]
                )

                # 4. Append
                chunk_data["beauty"].append(beauty)
                chunk_data["noisy"].append(noisy)
                chunk_data["albedo"].append(albedo)
                chunk_data["normal"].append(normal)
                chunk_data["pos"].append(pos)
                chunk_data["flow"].append(flow)
                valid_count += 1

                # 5. Save Chunk
                if len(chunk_data["beauty"]) >= CHUNK_SIZE:
                    save_chunk(chunk_data, scene, chunk_idx)
                    chunk_data = {k: [] for k in chunk_data}
                    chunk_idx += 1

            except Exception as e:
                print(f"Frame error: {e}")

        # Save leftovers
        if len(chunk_data["beauty"]) > 0:
            save_chunk(chunk_data, scene, chunk_idx)

        print(f"  -> Converted {valid_count} frames.")


if __name__ == "__main__":
    process_dataset()
