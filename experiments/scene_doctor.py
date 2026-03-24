import os
import re

SCENE_ROOT = "my_xml_scenes"


def diagnose():
    print(f"--- DIAGNOSING {SCENE_ROOT} ---\n")

    if not os.path.exists(SCENE_ROOT):
        print(f"CRITICAL: Folder '{SCENE_ROOT}' does not exist!")
        return

    folders = [f for f in os.scandir(SCENE_ROOT) if f.is_dir()]
    print(f"Found {len(folders)} folders.\n")

    for f in folders:
        print(f"Checking: [{f.name}]")

        # 1. CHECK FOR XML
        # We look for ANY .xml file to see if the naming is wrong
        xml_files = [x for x in os.listdir(f.path) if x.endswith(".xml")]

        if not xml_files:
            print(f"  ❌ FAIL: No .xml files found. (Is it just an OBJ/PLY model?)")
            continue

        # Check if the standard ones exist
        target_xml = None
        if "scene_v3.xml" in xml_files:
            target_xml = "scene_v3.xml"
        elif "scene.xml" in xml_files:
            target_xml = "scene.xml"

        if not target_xml:
            print(
                f"  ⚠️ WARN: Found XMLs {xml_files}, but script looks for 'scene.xml' or 'scene_v3.xml'."
            )
            print(f"     -> FIX: Rename '{xml_files[0]}' to 'scene.xml'")
            continue

        print(f"  ✅ XML Found: {target_xml}")

        # 2. CHECK FOR CAMERA
        with open(os.path.join(f.path, target_xml), "r", encoding="utf-8") as file:
            content = file.read()

        has_lookat = re.search(r"<lookat\s+origin=", content)
        has_matrix = re.search(r"<matrix\s+value=", content)

        if has_lookat:
            print("  ✅ Camera: Found <lookat> tag.")
        elif has_matrix:
            print("  ✅ Camera: Found <matrix> tag.")
        else:
            print("  ❌ FAIL: Could not find Camera (<lookat> or <matrix>).")
            print("     -> REASON: The regex couldn't find the camera coordinates.")
            print(
                "     -> FIX: The camera might be using 'transform' properties. We need to manually add a LookAt."
            )

        print("-" * 30)


if __name__ == "__main__":
    diagnose()
