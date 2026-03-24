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
OUTPUT_ROOT = "dataset_indoor_split"
FRAMES_PER_SCENE = 50
SPP_GT = 512
SPP_NOISY = 2
RES_X, RES_Y = 512, 512
DOLLY_SPEED = 0.5
PAN_SPEED = 0.1


def parse_camera_pos_from_text(xml_path):
    with open(xml_path, "r", encoding="utf-8") as f:
        content = f.read()

    # LookAt Pattern
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

    # Matrix Pattern
    matrix = re.search(r'<matrix\s+value="([^"]+)"', content)
    if matrix:
        try:
            nums = [float(x) for x in matrix.group(1).split()]
            m = np.array(nums).reshape(4, 4)
            fwd = m[:3, 2]
            up = m[:3, 1]
            return m[:3, 3], fwd / np.linalg.norm(fwd), up / np.linalg.norm(up)
        except:
            pass

    return None, None, None


def remove_tag_block(content, tag_name):
    """
    Surgically removes a tag and its children by counting nesting depth.
    Handles <integrator> ... <integrator> ... </integrator> ... </integrator> correctly.
    """
    while True:
        # 1. Find the start of the tag
        start_pattern = f"<{tag_name}"
        start_idx = content.find(start_pattern)

        # If not found, we are done
        if start_idx == -1:
            break

        # 2. Check if it's a self-closing tag immediately (e.g. <sensor ... />)
        # We search for the next '>' to check strict syntax
        close_bracket_idx = content.find(">", start_idx)
        if content[close_bracket_idx - 1] == "/":
            # It's self-closing <sensor ... />. Cut it out.
            content = content[:start_idx] + content[close_bracket_idx + 1 :]
            continue

        # 3. It is an open block. We must find the MATCHING closing tag.
        # We iterate through the string counting depth.
        depth = 1
        current_idx = start_idx + len(start_pattern)

        end_idx = -1

        while depth > 0 and current_idx < len(content):
            # Find next interesting symbol
            next_open = content.find(f"<{tag_name}", current_idx)
            next_close = content.find(f"</{tag_name}>", current_idx)

            # If no more tags, file is broken, break loop
            if next_close == -1:
                break

            # If we find an opener before a closer, depth increases
            if next_open != -1 and next_open < next_close:
                depth += 1
                current_idx = next_open + len(start_pattern)
            else:
                # We found a closer
                depth -= 1
                current_idx = next_close + len(tag_name) + 3  # +3 for </ and >
                if depth == 0:
                    end_idx = current_idx

        if end_idx != -1:
            # Cut the block out
            content = content[:start_idx] + content[end_idx:]
        else:
            # Failsafe: If we couldn't parse structure, break to avoid infinite loop
            print(
                f"  [Warn] Could not parse nesting for {tag_name}, falling back to simple regex."
            )
            # Fallback to simple regex deletion for this specific tag instance
            content = re.sub(rf"<{tag_name}[\s\S]*?</{tag_name}>", "", content, count=1)
            break

    return content


def create_patched_xml(original_xml, new_sensor, new_integrator, file_map):
    content = original_xml

    # 1. SURGICAL REMOVAL (Clean & Nested-Safe)
    content = remove_tag_block(content, "sensor")
    content = remove_tag_block(content, "integrator")

    # 2. ABSOLUTE PATH FIX
    def path_replacer(match):
        filename = match.group(0).split('"')[1]
        basename = os.path.basename(filename)
        if basename in file_map:
            # Force forward slashes
            return f'value="{file_map[basename].replace(os.sep, "/")}"'
        return match.group(0)

    content = re.sub(
        r'value="[^"]+\.(jpg|jpeg|png|tga|bmp|exr|hdr)"',
        path_replacer,
        content,
        flags=re.IGNORECASE,
    )

    # 3. INJECT NEW BLOCKS
    return content.replace("</scene>", f"\n{new_integrator}\n{new_sensor}\n</scene>")


def process_scenes():
    scene_folders = [f.path for f in os.scandir(SCENE_ROOT) if f.is_dir()]
    if not scene_folders:
        print("No scenes found.")
        return

    for folder in scene_folders:
        target_xml = os.path.join(folder, "scene_v3.xml")
        if not os.path.exists(target_xml):
            target_xml = os.path.join(folder, "scene.xml")
        if not os.path.exists(target_xml):
            continue

        scene_name = os.path.basename(folder)
        print(f"\n--- Processing: {scene_name} ---")

        file_map = {}
        for root, _, files in os.walk(folder):
            for file in files:
                if file.lower().endswith((".jpg", ".png", ".exr")):
                    file_map[file] = os.path.join(root, file)

        start_pos, start_fwd, start_up = parse_camera_pos_from_text(target_xml)
        if start_pos is None:
            continue

        with open(target_xml, "r", encoding="utf-8") as f:
            original_xml = f.read()

        out_dir = os.path.join(OUTPUT_ROOT, scene_name)
        os.makedirs(out_dir, exist_ok=True)

        for i in range(FRAMES_PER_SCENE):
            print(f"  Frame {i + 1}/{FRAMES_PER_SCENE}...", end="\r")

            t = i / FRAMES_PER_SCENE
            offset = start_pos + start_fwd * (np.sin(t * np.pi) * DOLLY_SPEED)
            pan = np.sin(t * 2 * np.pi) * PAN_SPEED
            origin = offset
            target = start_pos + start_fwd * 5.0
            target[0] += pan

            integrator = """
            <integrator type="aov">
                <string name="aovs" value="albedo:albedo,nn:sh_normal"/>
                <integrator type="path">
                    <integer name="max_depth" value="12"/>
                </integrator>
            </integrator>
            """

            sensor = f"""
            <sensor type="perspective">
                <string name="fov_axis" value="smaller"/>
                <float name="fov" value="45"/>
                <transform name="to_world">
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

                # --- RENDER GT ---
                tensor_all = mi.render(scene, spp=SPP_GT)

                # SPLIT CHANNELS
                beauty = tensor_all[:, :, 0:3]
                albedo = tensor_all[:, :, 3:6]
                normal = tensor_all[:, :, 6:9]

                mi.util.write_bitmap(f"{out_dir}/frame_{i:04d}_beauty.exr", beauty)
                mi.util.write_bitmap(f"{out_dir}/frame_{i:04d}_albedo.exr", albedo)
                mi.util.write_bitmap(f"{out_dir}/frame_{i:04d}_normal.exr", normal)

                # --- RENDER NOISY ---
                tensor_noisy = mi.render(scene, spp=SPP_NOISY)
                noisy_rgb = tensor_noisy[:, :, 0:3]
                mi.util.write_bitmap(f"{out_dir}/frame_{i:04d}_noisy.exr", noisy_rgb)

                c_mat = scene.sensors()[0].world_transform().matrix.numpy()
                np.save(f"{out_dir}/frame_{i:04d}_cam.npy", c_mat)

            except Exception as e:
                print(f"\n  Failed frame {i}: {e}")
            finally:
                if os.path.exists(temp_path):
                    os.remove(temp_path)


if __name__ == "__main__":
    process_scenes()
