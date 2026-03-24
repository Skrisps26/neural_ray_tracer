import glob
import os

import mitsuba as mi
import numpy as np
import torch
from tqdm import tqdm

# --- CONFIG ---
INPUT_ROOT = "dataset_indoor_safe"
OUTPUT_ROOT = "dataset_optimized"
CHUNK_SIZE = 50

# Set Mitsuba to Scalar mode (CPU) for loading files
try:
    mi.set_variant("scalar_rgb")
except:
    pass


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

    view_mat = np.linalg.inv(cam_next_mat)
    full_mat = view_mat.T @ proj_mat.T
    clip_space = points_4d @ full_mat

    w = clip_space[:, 3:4]
    w[np.abs(w) < 1e-5] = 1e-5
    ndc = clip_space[:, :3] / w

    screen_x = ((ndc[:, 0] + 1) * 0.5) * width
    screen_y = ((1 - ndc[:, 1]) * 0.5) * height

    grid_y, grid_x = np.mgrid[0:H, 0:W]
    flow = np.stack(
        [screen_x - grid_x.flatten(), screen_y - grid_y.flatten()], axis=1
    ).reshape(H, W, 2)

    mask = (w.flatten() < 0).reshape(H, W)
    flow[mask] = 0
    return flow


def load_exr_safe(path):
    """
    Uses Mitsuba to load EXR.
    Returns: Numpy array (H, W, 3) or None if corrupt.
    """
    try:
        # 1. Size Check: If file is < 500KB, it's garbage.
        if os.path.getsize(path) < 500 * 1024:
            return None

        # 2. Load with Mitsuba
        bmp = mi.Bitmap(path)
        arr = np.array(bmp)

        # 3. Shape Check
        if arr.shape[0] != 512 or arr.shape[1] != 512:
            return None

        return arr[:, :, :3]  # Drop Alpha if exists

    except Exception:
        return None


def save_chunk(data, scene, idx):
    if len(data["beauty"]) == 0:
        return

    def to_tensor(params):
        arr = np.stack(params)
        return torch.from_numpy(arr).permute(0, 3, 1, 2).half()

    out_dict = {
        "beauty": to_tensor(data["beauty"]),
        "noisy": to_tensor(data["noisy"]),
        "albedo": to_tensor(data["albedo"]),
        "normal": to_tensor(data["normal"]),
        "pos": to_tensor(data["pos"]),
        "flow": to_tensor(data["flow"]),
    }

    save_path = os.path.join(OUTPUT_ROOT, f"{scene}_chunk_{idx:02d}.pt")
    torch.save(out_dict, save_path)
    print(f"  [Saved] {save_path} ({len(data['beauty'])} frames)")


def process_dataset():
    scenes = [
        d for d in os.listdir(INPUT_ROOT) if os.path.isdir(os.path.join(INPUT_ROOT, d))
    ]
    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    PROJ_MAT = get_projection_matrix(fov=60, aspect=1.0)

    for scene in scenes:
        print(f"\n--- Processing: {scene} ---")
        scene_path = os.path.join(INPUT_ROOT, scene)
        files = sorted(glob.glob(os.path.join(scene_path, "*_beauty.exr")))

        chunk_data = {
            "beauty": [],
            "noisy": [],
            "albedo": [],
            "normal": [],
            "pos": [],
            "flow": [],
        }
        chunk_idx = 0
        skipped_corrupt = 0

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

            # Check Existence
            if not all(os.path.exists(p) for p in paths.values()):
                continue

            # Check Integrity & Load
            beauty = load_exr_safe(f_beauty)
            noisy = load_exr_safe(paths["noisy"])
            albedo = load_exr_safe(paths["albedo"])
            normal = load_exr_safe(paths["normal"])
            pos = load_exr_safe(paths["pos"])

            # If ANY file in the set is bad, skip the whole frame
            if any(x is None for x in [beauty, noisy, albedo, normal, pos]):
                skipped_corrupt += 1
                continue

            try:
                cam_curr = np.load(paths["cam"])
                cam_next = np.load(paths["cam_next"])

                flow = compute_flow(
                    pos, cam_next, PROJ_MAT, beauty.shape[1], beauty.shape[0]
                )

                chunk_data["beauty"].append(beauty)
                chunk_data["noisy"].append(noisy)
                chunk_data["albedo"].append(albedo)
                chunk_data["normal"].append(normal)
                chunk_data["pos"].append(pos)
                chunk_data["flow"].append(flow)

                if len(chunk_data["beauty"]) >= CHUNK_SIZE:
                    save_chunk(chunk_data, scene, chunk_idx)
                    chunk_data = {k: [] for k in chunk_data}
                    chunk_idx += 1

            except Exception as e:
                print(f"Error computing flow: {e}")

        # Save leftovers
        if len(chunk_data["beauty"]) > 0:
            save_chunk(chunk_data, scene, chunk_idx)

        if skipped_corrupt > 0:
            print(
                f"  [Warning] Skipped {skipped_corrupt} corrupt/truncated frames in {scene}"
            )


if __name__ == "__main__":
    process_dataset()
