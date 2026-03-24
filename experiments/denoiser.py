import glob
import os

import drjit as dr
import mitsuba as mi
import numpy as np

# Force GPU
try:
    mi.set_variant("cuda_ad_rgb")
except:
    print("Error: You need a GPU (cuda_ad_rgb) to use the Denoiser!")
    exit()

# --- CONFIG ---
DATASET_DIR = "dataset_indoor_realism"


def polish_dataset():
    gt_files = glob.glob(os.path.join(DATASET_DIR, "**", "*_gt.exr"), recursive=True)

    if not gt_files:
        print(f"No GT files found in {DATASET_DIR}")
        return

    print(f"Found {len(gt_files)} frames to polish...")

    denoiser = None
    current_size_tuple = None

    for i, filepath in enumerate(gt_files):
        print(
            f"Polishing {i + 1}/{len(gt_files)}: {os.path.basename(filepath)}...",
            end="\r",
        )

        try:
            bitmap = mi.Bitmap(filepath)

            # --- THE FIX: Convert DrJit vector to Python Tuple ---
            # This ensures '!=' comparison works like standard Python
            size_vec = bitmap.size()
            size_tuple = (size_vec.x, size_vec.y)

            # Re-initialize denoiser if image size changes
            if denoiser is None or size_tuple != current_size_tuple:
                # We pass the vector to Mitsuba, but store the tuple for comparison
                denoiser = mi.OptixDenoiser(size_vec, albedo=True, normals=True)
                current_size_tuple = size_tuple

            # Extract Channels
            channels = dict(bitmap.split())

            # Helper: Stack individual R, G, B bitmaps into a Tensor
            def merge_to_tensor(ch_r, ch_g, ch_b):
                r = np.array(channels[ch_r])
                g = np.array(channels[ch_g])
                b = np.array(channels[ch_b])
                # dstack makes it (H, W, 3)
                return mi.TensorXf(np.dstack((r, g, b)))

            # Check for required AOVs
            # Note: channel names might vary slightly depending on Mitsuba version
            # We try standard names first
            if "albedo.R" in channels:
                img_albedo = merge_to_tensor("albedo.R", "albedo.G", "albedo.B")
            elif "albedo.r" in channels:  # Sometimes lowercase
                img_albedo = merge_to_tensor("albedo.r", "albedo.g", "albedo.b")
            else:
                print(f"\n[Skip] {filepath} missing Albedo.")
                continue

            if "nn.X" in channels:
                img_normal = merge_to_tensor("nn.X", "nn.Y", "nn.Z")
            elif "nn.x" in channels:
                img_normal = merge_to_tensor("nn.x", "nn.y", "nn.z")
            else:
                print(f"\n[Skip] {filepath} missing Normals.")
                continue

            # Beauty is usually just R, G, B
            img_beauty = merge_to_tensor("R", "G", "B")

            # RUN DENOISER
            denoised_tensor = denoiser(
                img_beauty, albedo=img_albedo, normals=img_normal
            )

            # Save as _clean.exr
            clean_path = filepath.replace("_gt.exr", "_gt_clean.exr")
            mi.util.write_bitmap(clean_path, denoised_tensor)

        except Exception as e:
            # Print the error but keep going so one bad file doesn't kill the process
            print(f"\n[Error] Failed on {filepath}: {e}")
            continue

    print("\n\nPolishing Complete!")


if __name__ == "__main__":
    polish_dataset()
