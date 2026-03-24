import glob
import os
import re

import drjit as dr
import mitsuba as mi
import numpy as np

# --- 1. SETUP GPU ---
try:
    mi.set_variant("cuda_ad_rgb")
    print("Success! Running on GPU.")
except:
    print("GPU failed. Falling back to CPU.")
    mi.set_variant("scalar_rgb")

# --- CONFIG ---
SCENE_ROOT = "my_xml_scenes"
OUTPUT_ROOT = "dataset_indoor_safe"  # New safe folder
FRAMES_PER_SCENE = 200  # Plenty of data
RES_X, RES_Y = 512, 512

# --- SPEED SETTINGS ---
# We render fast (64 SPP) and use AI to clean it up instantly.
SPP_GT_RAW = 256  # Fast!
SPP_NOISY = 2  # Input


def parse_camera_pos_from_text(xml_path):
    with open(xml_path, "r", encoding="utf-8") as f:
        content = f.read()

    # LookAt
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

    # Matrix
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


def create_patched_xml(original_xml, new_sensor, new_integrator, file_map):
    content = original_xml

    # --- SANITIZER 2.0 ---
    # Fix 1: Snake Case
    content = content.replace('name="toWorld"', 'name="to_world"')

    # Fix 2: Nuke broken Hair materials
    if 'type="roughhair"' in content:
        # Regex to find and replace the whole hair block with a safe diffuse brown
        clean_block = '<bsdf type="diffuse"><rgb name="reflectance" value="0.4, 0.25, 0.1"/></bsdf>'
        content = re.sub(
            r'<bsdf[^>]*type="roughhair"[^>]*>[\s\S]*?</bsdf>', clean_block, content
        )
        content = content.replace('<bsdf type="roughhair"/>', clean_block)

    # --- CLEAN REMOVAL ---
    # Robustly remove old sensors/integrators
    for tag in ["sensor", "integrator"]:
        while True:
            start = content.find(f"<{tag}")
            if start == -1:
                break
            # Check for self-closing
            close_bracket = content.find(">", start)
            if content[close_bracket - 1] == "/":
                content = content[:start] + content[close_bracket + 1 :]
                continue
            # Check for full block (simple regex fallback for speed/safety here)
            content = re.sub(rf"<{tag}[\s\S]*?</{tag}>", "", content, count=1)

    # --- PATH FIX ---
    def path_replacer(match):
        filename = match.group(0).split('"')[1]
        basename = os.path.basename(filename)
        if basename in file_map:
            return f'value="{file_map[basename].replace(os.sep, "/")}"'
        return match.group(0)

    content = re.sub(
        r'value="[^"]+\.(jpg|jpeg|png|tga|bmp|exr|hdr)"',
        path_replacer,
        content,
        flags=re.IGNORECASE,
    )

    return content.replace("</scene>", f"\n{new_integrator}\n{new_sensor}\n</scene>")


def save_f16(path, tensor):
    """Save as Float16 to save 50% disk space"""
    bmp = mi.Bitmap(tensor)
    bmp = bmp.convert(
        mi.Bitmap.PixelFormat.RGB, mi.Struct.Type.Float16, srgb_gamma=False
    )
    bmp.write(path)


def process_scenes():
    scene_folders = [f.path for f in os.scandir(SCENE_ROOT) if f.is_dir()]

    for folder in scene_folders:
        scene_name = os.path.basename(folder)
        target_xml = os.path.join(folder, "scene_v3.xml")
        if not os.path.exists(target_xml):
            target_xml = os.path.join(folder, "scene.xml")
        if not os.path.exists(target_xml):
            continue

        print(f"\n--- Processing: {scene_name} ---")

        # Map Assets
        file_map = {}
        for root, _, files in os.walk(folder):
            for file in files:
                if file.lower().endswith((".jpg", ".png", ".exr")):
                    file_map[file] = os.path.join(root, file)

        # Get Safe Start Position
        start_pos, start_fwd, start_up = parse_camera_pos_from_text(target_xml)
        if start_pos is None:
            print("  [Skip] No camera found.")
            continue

        start_right = np.cross(start_fwd, start_up)

        with open(target_xml, "r", encoding="utf-8") as f:
            original_xml = f.read()

        out_dir = os.path.join(OUTPUT_ROOT, scene_name)
        os.makedirs(out_dir, exist_ok=True)

        # --- INITIALIZE DENOISER (Reuse for speed) ---
        denoiser = mi.OptixDenoiser(
            input_size=(RES_X, RES_Y), albedo=True, normals=True
        )

        for i in range(FRAMES_PER_SCENE):
            print(f"  Frame {i + 1}/{FRAMES_PER_SCENE}...", end="\r")
            t = i / FRAMES_PER_SCENE

            # --- SAFE "HEAD TURN" MOVEMENT ---
            # 1. Position: Stay mostly still (just 5cm breathing wobble)
            wobble_x = np.sin(t * 13.0) * 0.05
            wobble_y = np.cos(t * 17.0) * 0.05
            wobble_z = np.sin(t * 19.0) * 0.02

            origin = (
                start_pos
                + (start_right * wobble_x)
                + (start_fwd * wobble_y)
                + (start_up * wobble_z)
            )

            # 2. Rotation: Look around the room (The "Security Camera" sweep)
            # Yaw (Left/Right)
            yaw = np.sin(t * 2 * np.pi) * 1.5
            # Pitch (Up/Down) - looking at floor/ceiling
            pitch = np.cos(t * 4 * np.pi) * 0.4

            # Calculate target
            target_offset = (start_fwd * 2.0) + (start_right * yaw) + (start_up * pitch)
            target = origin + target_offset

            # --- XML BLOCKS ---
            # Added 'pos' for future Lagrangian model
            integrator = """
            <integrator type="aov">
                <string name="aovs" value="albedo:albedo,nn:sh_normal,pos:position"/>
                <integrator type="path">
                    <integer name="max_depth" value="6"/>
                </integrator>
            </integrator>
            """

            sensor = f"""
            <sensor type="perspective">
                <string name="fov_axis" value="smaller"/>
                <float name="fov" value="60"/> <transform name="to_world">
                    <lookat origin="{origin[0]:.4f}, {origin[1]:.4f}, {origin[2]:.4f}"
                            target="{target[0]:.4f}, {target[1]:.4f}, {target[2]:.4f}"
                            up="{start_up[0]:.4f}, {start_up[1]:.4f}, {start_up[2]:.4f}"/>
                </transform>
                <film type="hdrfilm">
                    <integer name="width" value="{RES_X}"/>
                    <integer name="height" value="{RES_Y}"/>
                    <string name="pixel_format" value="rgb"/>
                    <rfilter type="box"/>
                </film>
            </sensor>
            """

            patched = create_patched_xml(original_xml, sensor, integrator, file_map)
            temp_path = os.path.join(folder, "_temp_render.xml")
            with open(temp_path, "w", encoding="utf-8") as f:
                f.write(patched)

            try:
                scene = mi.load_file(temp_path)

                # --- FAST RENDER + DENOISE ---
                tensor_all = mi.render(scene, spp=SPP_GT_RAW)  # Fast 64 SPP

                # Slice (Beauty, Albedo, Normal, Position)
                beauty_noisy = tensor_all[:, :, 0:3]
                albedo = tensor_all[:, :, 3:6]
                normal = tensor_all[:, :, 6:9]
                position = tensor_all[:, :, 9:12]  # Need this for Advection later!

                # Instant Denoise (Make 64 SPP look like 1024 SPP)
                beauty_clean = denoiser(beauty_noisy, albedo=albedo, normals=normal)

                # Save Safe (Float16)
                save_f16(f"{out_dir}/frame_{i:04d}_beauty.exr", beauty_clean)
                save_f16(f"{out_dir}/frame_{i:04d}_albedo.exr", albedo)
                save_f16(f"{out_dir}/frame_{i:04d}_normal.exr", normal)
                save_f16(f"{out_dir}/frame_{i:04d}_pos.exr", position)

                # Input Noisy
                tensor_input = mi.render(scene, spp=SPP_NOISY)
                save_f16(f"{out_dir}/frame_{i:04d}_noisy.exr", tensor_input[:, :, 0:3])

            except Exception as e:
                print(f"  Failed frame {i}: {e}")
            finally:
                if os.path.exists(temp_path):
                    os.remove(temp_path)


if __name__ == "__main__":
    process_scenes()
