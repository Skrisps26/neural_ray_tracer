import os
import re

# --- CONFIG ---
SCENE_FOLDER = "my_xml_scenes/water-caustic"
# If your file is named scene.xml, leave this. If it's something else, change it.
INPUT_FILE = "scene_v0.6.xml"


def upgrade_v06():
    full_path = os.path.join(SCENE_FOLDER, INPUT_FILE)
    if not os.path.exists(full_path):
        # Try to auto-detect
        possible = [f for f in os.listdir(SCENE_FOLDER) if f.endswith(".xml")]
        if possible:
            full_path = os.path.join(SCENE_FOLDER, possible[0])
            print(f"Auto-detected file: {possible[0]}")
        else:
            print(f"Error: No XML found in {SCENE_FOLDER}")
            return

    with open(full_path, "r", encoding="utf-8") as f:
        content = f.read()

    print(f"Upgrading {os.path.basename(full_path)}...")

    # --- 1. TAG RENAMES ---
    # v0.6 used <camera>, <luminaire>. v3 uses <sensor>, <emitter>.
    content = content.replace("<camera", "<sensor")
    content = content.replace("</camera>", "</sensor>")
    content = content.replace("<luminaire", "<emitter")
    content = content.replace("</luminaire>", "</emitter>")

    # --- 2. PARAMETER RENAMES (CamelCase -> snake_case) ---
    replacements = {
        "intIOR": "int_ior",
        "extIOR": "ext_ior",
        "pixelFormat": "pixel_format",
        "shutterOpen": "shutter_open",
        "shutterClose": "shutter_close",
        "faceNormals": "face_normals",
        "uvscale": "uv_scale",
        "componentFormat": "component_format",
    }

    for old, new in replacements.items():
        content = content.replace(f'name="{old}"', f'name="{new}"')

    # --- 3. VERSION BUMP ---
    content = re.sub(r'version="\d+\.\d+\.\d+"', 'version="3.0.0"', content)

    # --- 4. CAUSTICS INJECTION ---
    # We inject a powerful Sun and a Safety Floor before the closing </scene> tag.
    # This ensures that even if the original lighting is broken/dim, you get caustics.

    injection = """
    <shape type="sphere">
        <point name="center" x="2" y="10" z="5"/>
        <float name="radius" value="0.1"/>
        <emitter type="area">
            <rgb name="radiance" value="1000.0"/>
        </emitter>
    </shape>

    <shape type="rectangle">
        <transform name="to_world">
            <scale value="20"/>
            <rotate x="1" angle="-90"/>
            <translate y="-5"/>
        </transform>
        <bsdf type="diffuse">
            <rgb name="reflectance" value="0.5, 0.5, 0.5"/>
        </bsdf>
    </shape>
    """

    if "</scene>" in content:
        content = content.replace("</scene>", f"{injection}\n</scene>")

    # --- 5. SAVE ---
    output_path = os.path.join(SCENE_FOLDER, "scene_v3.xml")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(content)

    print(f"Success! Saved to: {output_path}")
    print("Run 'generate_varied_dataset.py' to render.")


if __name__ == "__main__":
    upgrade_v06()
