import glob
import os

import mitsuba as mi
import numpy as np

# --- CONFIG ---
DATASET_DIR = "dataset_indoor_safe"
VARIANCE_THRESHOLD = 0.01  # If contrast is lower than this, it's a flat color
BRIGHTNESS_THRESHOLD = 0.01  # If darker than this, it's pitch black


def clean_dataset():
    print(f"Scanning {DATASET_DIR} for bad frames...")

    # Check all beauty frames
    files = glob.glob(os.path.join(DATASET_DIR, "**", "*_beauty.exr"), recursive=True)
    deleted_count = 0

    for i, f_beauty in enumerate(files):
        print(f"Checking {i + 1}/{len(files)}...", end="\r")

        try:
            # Load and convert to numpy
            bmp = mi.Bitmap(f_beauty)
            img = np.array(bmp)  # Shape (512, 512, 3)

            # 1. Check if Pitch Black
            max_val = np.max(img)
            if max_val < BRIGHTNESS_THRESHOLD:
                print(
                    f"\n[Delete] {os.path.basename(f_beauty)} (Too Dark: {max_val:.5f})"
                )
                delete_frame_set(f_beauty)
                deleted_count += 1
                continue

            # 2. Check if Solid Color (White/Gray Wall)
            # We calculate Standard Deviation (Contrast)
            std_val = np.std(img)
            if std_val < VARIANCE_THRESHOLD:
                print(
                    f"\n[Delete] {os.path.basename(f_beauty)} (Flat Color: {std_val:.5f})"
                )
                delete_frame_set(f_beauty)
                deleted_count += 1
                continue

        except Exception as e:
            print(f"\n[Error] Could not check {f_beauty}: {e}")

    print(f"\n\nDone! Deleted {deleted_count} bad frames.")


def delete_frame_set(f_beauty):
    """
    Deletes the Beauty, Albedo, Normal, Noisy, and Cam files for a specific frame.
    """
    base_path = f_beauty.replace("_beauty.exr", "")

    suffixes = ["_beauty.exr", "_albedo.exr", "_normal.exr", "_noisy.exr", "_cam.npy"]

    for suffix in suffixes:
        path = base_path + suffix
        if os.path.exists(path):
            os.remove(path)


if __name__ == "__main__":
    clean_dataset()
