import glob
import os
import re

import drjit as dr
import mitsuba as mi
import numpy as np

# Set variant (Scalar is safer for complex XMLs if CUDA fails, but CUDA is faster)
try:
    mi.set_variant("scalar_rgb")
except:
    mi.set_variant("scalar_rgb")

# --- CONFIG ---
SCENE_ROOT = "my_xml_scenes"
OUTPUT_ROOT = "dataset_xml_realism"
FRAMES_PER_SCENE = 40
SPP_GT = 512
SPP_NOISY = 2
RES_X, RES_Y = 512, 512


def get_scene_center_safe(xml_path):
    """
    Loads the scene simply to calculate the bounding box.
    """
    try:
        # We append the folder to the resolver so Mitsuba finds textures
        mi.Thread.thread().file_resolver().append(os.path.dirname(xml_path))
        scene = mi.load_file(xml_path)
        bbox = scene.bbox()
        return bbox.center(), dr.max(bbox.max - bbox.min)[0]
    except Exception as e:
        print(f"  [Warn] BBox detection failed: {e}. Using defaults.")
        return mi.Point3f(0, 0, 0), 5.0


def create_patched_xml(original_xml_content, new_sensor_xml, new_integrator_xml):
    """
    Removes old sensor/integrator tags and appends new ones.
    """
    # 1. Remove existing Sensor and Integrator using Regex
    # Matches <sensor ...> ... </sensor> OR <sensor ... />
    # DOTALL makes (.) match newlines
    content = re.sub(r"<sensor[\s\S]*?</sensor>", "", original_xml_content)
    content = re.sub(r"<sensor[\s\S]*?/>", "", content)

    content = re.sub(r"<integrator[\s\S]*?</integrator>", "", content)
    content = re.sub(r"<integrator[\s\S]*?/>", "", content)

    # 2. Inject new blocks before the closing </scene> tag
    new_content = content.replace(
        "</scene>", f"\n{new_integrator_xml}\n{new_sensor_xml}\n</scene>"
    )

    return new_content


def process_xml_scenes():
    xml_files = glob.glob(os.path.join(SCENE_ROOT, "**/*.xml"), recursive=True)

    if not xml_files:
        print(f"Error: No .xml files found in {SCENE_ROOT}")
        return

    print(f"Found {len(xml_files)} scenes.")

    for xml_path in xml_files:
        scene_name = os.path.basename(os.path.dirname(xml_path))
        print(f"\n--- Processing: {scene_name} ---")

        # 1. Setup resolver so textures load
        scene_dir = os.path.dirname(xml_path)
        mi.Thread.thread().file_resolver().append(scene_dir)

        # 2. Get Geometry Info
        center_gpu, size_gpu = get_scene_center_safe(xml_path)

        # Move to CPU for calculations
        cx, cy, cz = float(center_gpu.x), float(center_gpu.y), float(center_gpu.z)
        size = float(size_gpu[0]) if hasattr(size_gpu, "__len__") else float(size_gpu)

        print(f"  Center: [{cx:.2f}, {cy:.2f}, {cz:.2f}] | Size: {size:.2f}")

        # 3. Read Original XML
        with open(xml_path, "r") as f:
            original_xml = f.read()

        out_dir = os.path.join(OUTPUT_ROOT, scene_name)
        os.makedirs(out_dir, exist_ok=True)

        # 4. Render Loop
        for i in range(FRAMES_PER_SCENE):
            print(f"  Frame {i + 1}/{FRAMES_PER_SCENE}...", end="\r")

            # Orbit Math
            t = i / FRAMES_PER_SCENE
            angle = t * 2 * np.pi
            radius = size * 0.8
            cam_x = cx + np.sin(angle) * radius
            cam_y = cy - np.cos(angle) * radius
            cam_z = cz + size * 0.2

            # 5. Define New Components as XML Strings
            # We use f-strings to inject the calculated camera coordinates

            new_integrator = """
            <integrator type="aov">
                <string name="aovs" value="albedo:albedo,nn:sh_normal,pos:position"/>
                <integrator type="path">
                    <integer name="max_depth" value="8"/>
                </integrator>
            </integrator>
            """

            new_sensor = f"""
            <sensor type="perspective">
                <string name="fov_axis" value="smaller"/>
                <float name="fov" value="45"/>
                <transform name="to_world">
                    <lookat origin="{cam_x}, {cam_y}, {cam_z}"
                            target="{cx}, {cy}, {cz}"
                            up="0, 0, 1"/>
                </transform>
                <film type="hdrfilm">
                    <integer name="width" value="{RES_X}"/>
                    <integer name="height" value="{RES_Y}"/>
                    <string name="pixel_format" value="rgb"/>
                </film>
            </sensor>
            """

            # 6. Patch and Load
            final_xml = create_patched_xml(original_xml, new_sensor, new_integrator)

            try:
                # load_string parses the XML text we just created
                scene = mi.load_string(final_xml)

                # Render GT
                img_gt = mi.render(scene, spp=SPP_GT)
                mi.util.write_bitmap(f"{out_dir}/frame_{i:04d}_gt.exr", img_gt)

                # Render Noisy
                img_noisy = mi.render(scene, spp=SPP_NOISY)
                mi.util.write_bitmap(f"{out_dir}/frame_{i:04d}_noisy.exr", img_noisy)

                # Save Cam
                c_mat = scene.sensors()[0].world_transform().matrix.numpy()
                np.save(f"{out_dir}/frame_{i:04d}_cam.npy", c_mat)

            except Exception as e:
                print(f"\n  Failed frame {i}: {e}")
                continue


if __name__ == "__main__":
    process_xml_scenes()
