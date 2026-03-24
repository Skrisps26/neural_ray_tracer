import glob
import os

import mitsuba as mi
import numpy as np

# --- CONFIG ---
INPUT_ROOT = "dataset_indoor_safe"


def check_scene():
    print(f"--- DIAGNOSTIC REPORT FOR: {INPUT_ROOT} ---")

    if not os.path.exists(INPUT_ROOT):
        print(f"❌ Root folder '{INPUT_ROOT}' does not exist.")
        return

    scenes = [
        d for d in os.listdir(INPUT_ROOT) if os.path.isdir(os.path.join(INPUT_ROOT, d))
    ]
    if not scenes:
        print("❌ No scene folders found.")
        return

    first_scene = scenes[0]
    scene_path = os.path.join(INPUT_ROOT, first_scene)
    print(f"Checking Scene: '{first_scene}'")

    # Check Frame 0000
    base_name = "frame_0000"
    extensions = {
        "Beauty": "_beauty.exr",
        "Noisy": "_noisy.exr",
        "Albedo": "_albedo.exr",
        "Normal": "_normal.exr",
        "Pos": "_pos.exr",
        "Cam": "_cam.npy",
    }

    all_good = True
    for name, ext in extensions.items():
        filename = base_name + ext
        full_path = os.path.join(scene_path, filename)

        if not os.path.exists(full_path):
            print(f"  ❌ MISSING: {filename}")
            all_good = False
        else:
            size_kb = os.path.getsize(full_path) / 1024
            print(f"  ✅ FOUND: {filename} ({size_kb:.1f} KB)")

            # Try loading EXR
            if ext.endswith(".exr"):
                try:
                    bmp = mi.Bitmap(full_path)
                    print(f"     -> Open Success: {bmp.size()}")
                except Exception as e:
                    print(f"     -> ❌ OPEN FAILED: {e}")
                    all_good = False

    if all_good:
        print("\nResult: Files look valid. The previous script was just too strict.")
    else:
        print("\nResult: Files are missing or corrupt.")


if __name__ == "__main__":
    check_scene()
