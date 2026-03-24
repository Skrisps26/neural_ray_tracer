import os
import re

# --- CONFIG ---
SCENE_FOLDER = "my_xml_scenes/furball/models"
# If your folder name is different, change it above.


def parse_old_camera(folder):
    """
    Scans for any .xml file in the folder to steal camera coordinates.
    """
    camera_data = None

    # Find the old scene file (could be scene.xml, scene_v0.6.xml, etc.)
    for f in os.listdir(folder):
        if f.endswith(".xml") and "v3" not in f:
            full_path = os.path.join(folder, f)
            with open(full_path, "r", encoding="utf-8") as file:
                content = file.read()

            # Regex for LookAt
            match = re.search(
                r'<lookat\s+origin="([^"]+)"\s+target="([^"]+)"\s+up="([^"]+)"', content
            )
            if match:
                camera_data = (match.group(1), match.group(2), match.group(3))
                print(f"  Camera found in {f}")
                break

    if not camera_data:
        print("  [Warning] No camera found. Using default view.")
        return "0, 0, 4", "0, 0, 0", "0, 1, 0"

    return camera_data


def rescue_hair():
    if not os.path.exists(SCENE_FOLDER):
        print(f"Error: Folder {SCENE_FOLDER} not found.")
        return

    # 1. FIND THE .MITSHAIR FILE
    hair_file = None
    for f in os.listdir(SCENE_FOLDER):
        if f.endswith(".mitshair"):
            hair_file = f
            break

    if not hair_file:
        print(f"Error: No .mitshair file found in {SCENE_FOLDER}")
        return

    print(f"Found Hair Asset: {hair_file}")

    # 2. GET CAMERA
    origin, target, up = parse_old_camera(SCENE_FOLDER)

    # 3. GENERATE V3 SCENE
    # We use the <shape type="hair"> plugin specifically for this format.
    # We also attach a 'roughhair' BSDF for realistic shine.

    xml_content = f"""
<scene version="3.0.0">
    <integrator type="path">
        <integer name="max_depth" value="12"/>
    </integrator>

    <sensor type="perspective">
        <string name="fov_axis" value="smaller"/>
        <float name="fov" value="45"/>
        <transform name="to_world">
            <lookat origin="{origin}" target="{target}" up="{up}"/>
        </transform>
        <film type="hdrfilm">
            <integer name="width" value="512"/>
            <integer name="height" value="512"/>
            <rfilter type="box"/>
        </film>
    </sensor>

    <shape type="sphere">
        <point name="center" x="2" y="5" z="5"/>
        <float name="radius" value="1.0"/>
        <emitter type="area">
            <rgb name="radiance" value="30.0"/>
        </emitter>
    </shape>

    <shape type="sphere">
        <point name="center" x="-2" y="2" z="-2"/>
        <float name="radius" value="0.5"/>
        <emitter type="area">
            <rgb name="radiance" value="5.0"/>
        </emitter>
    </shape>

    <shape type="hair">
        <string name="filename" value="{hair_file}"/>

        <bsdf type="roughhair">
            <rgb name="reflectance" value="0.4, 0.25, 0.1"/> <float name="melanin" value="0.5"/>
        </bsdf>
    </shape>
</scene>
"""

    # 4. SAVE
    output_path = os.path.join(SCENE_FOLDER, "scene_v3.xml")
    with open(output_path, "w") as f:
        f.write(xml_content)

    print(f"\nSuccess! Created {output_path}")
    print("Run 'generate_varied_dataset.py' to render it.")


if __name__ == "__main__":
    rescue_hair()
