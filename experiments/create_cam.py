import os
import re

import mitsuba as mi
import numpy as np

# --- CONFIG ---
# Must match your generator settings exactly!
SCENE_ROOT = "my_xml_scenes"
OUTPUT_ROOT = "dataset_indoor_safe"
FRAMES_PER_SCENE = 200

# Set variant to scalar (CPU) for speed, we don't need GPU for this
try:
    mi.set_variant("scalar_rgb")
except:
    pass


def parse_camera_pos_from_text(xml_path):
    """
    Extracts the initial camera state from the scene XML.
    Same logic as the generator.
    """
    with open(xml_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Try LookAt first
    lookat = re.search(
        r'<lookat\s+origin="([^"]+)"\s+target="([^"]+)"\s+up="([^"]+)"', content
    )
    if lookat:
        try:
            o = np.array([float(x) for x in lookat.group(1).replace(",", " ").split()])
            t = np.array([float(x) for x in lookat.group(2).replace(",", " ").split()])
            u = np.array([float(x) for x in lookat.group(3).replace(",", " ").split()])
            return o, (t - o) / np.linalg.norm(t - o), u / np.linalg.norm(u)
        except:
            pass

    # Try Matrix
    matrix = re.search(r'<matrix[^>]*value="([^"]+)"', content)
    if matrix:
        try:
            nums = [float(x) for x in matrix.group(1).replace(",", " ").split()]
            if len(nums) == 16:
                m = np.array(nums).reshape(4, 4)
                origin = m[:3, 3]
                fwd = m[:3, 2]
                up = m[:3, 1]
                if np.linalg.norm(fwd) == 0:
                    fwd = np.array([0, 0, 1])
                if np.linalg.norm(up) == 0:
                    up = np.array([0, 1, 0])
                return origin, fwd / np.linalg.norm(fwd), up / np.linalg.norm(up)
        except:
            pass

    return None, None, None


def recover_data():
    if not os.path.exists(SCENE_ROOT) or not os.path.exists(OUTPUT_ROOT):
        print("Error: Input or Output folder not found.")
        return

    # Get list of scenes in the OUTPUT folder (only fix scenes we actually rendered)
    rendered_scenes = [
        d
        for d in os.listdir(OUTPUT_ROOT)
        if os.path.isdir(os.path.join(OUTPUT_ROOT, d))
    ]

    print(f"Recovering camera data for {len(rendered_scenes)} scenes...")

    for scene_name in rendered_scenes:
        # Find original XML to get the start position
        scene_folder = os.path.join(SCENE_ROOT, scene_name)
        target_xml = os.path.join(scene_folder, "scene_v3.xml")
        if not os.path.exists(target_xml):
            target_xml = os.path.join(scene_folder, "scene.xml")

        if not os.path.exists(target_xml):
            print(f"  [Skip] Could not find original XML for {scene_name}")
            continue

        # Parse Start Pos
        start_pos, start_fwd, start_up = parse_camera_pos_from_text(target_xml)
        if start_pos is None:
            print(f"  [Skip] Could not parse camera in {scene_name}")
            continue

        start_right = np.cross(start_fwd, start_up)
        out_dir = os.path.join(OUTPUT_ROOT, scene_name)

        print(f"  Fixing {scene_name}...", end="\r")

        for i in range(FRAMES_PER_SCENE):
            t = i / FRAMES_PER_SCENE

            # --- REPLICATE THE EXACT MATH FROM GENERATOR ---

            # 1. Position Wobble
            wobble_x = np.sin(t * 13.0) * 0.05
            wobble_y = np.cos(t * 17.0) * 0.05
            wobble_z = np.sin(t * 19.0) * 0.02

            origin = (
                start_pos
                + (start_right * wobble_x)
                + (start_fwd * wobble_y)
                + (start_up * wobble_z)
            )

            # 2. Rotation Sweep
            yaw = np.sin(t * 2 * np.pi) * 1.5
            pitch = np.cos(t * 4 * np.pi) * 0.4

            target_offset = (start_fwd * 2.0) + (start_right * yaw) + (start_up * pitch)
            target = origin + target_offset

            # 3. Compute Matrix using Mitsuba
            # transform = lookat(origin, target, up)
            transform = mi.ScalarTransform4f.look_at(origin, target, start_up)

            # Extract 4x4 numpy matrix
            matrix = transform.matrix.numpy()

            # 4. Save
            np.save(f"{out_dir}/frame_{i:04d}_cam.npy", matrix)

    print("\nDone! Camera files restored.")


if __name__ == "__main__":
    recover_data()
