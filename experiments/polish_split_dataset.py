import glob
import os

import drjit as dr
import mitsuba as mi

# 1. Force GPU (Mandatory for OptiX Denoiser)
try:
    mi.set_variant("cuda_ad_rgb")
except Exception as e:
    print("Error: This script requires an NVIDIA GPU with 'cuda_ad_rgb'.")
    exit()

# --- CONFIG ---
DATASET_DIR = "dataset_indoor_split"


def polish():
    # Find all "beauty" files (the main render)
    files = glob.glob(os.path.join(DATASET_DIR, "**", "*_beauty.exr"), recursive=True)

    if not files:
        print(f"No '_beauty.exr' files found in {DATASET_DIR}")
        return

    print(f"Found {len(files)} frames to polish...")

    # Lazy init of denoiser
    denoiser = None
    current_size = None

    for i, f_beauty in enumerate(files):
        print(
            f"Polishing {i + 1}/{len(files)}: {os.path.basename(f_beauty)}...", end="\r"
        )

        # 1. Locate the helper files
        f_albedo = f_beauty.replace("_beauty.exr", "_albedo.exr")
        f_normal = f_beauty.replace("_beauty.exr", "_normal.exr")

        # 2. Verify they exist
        if not os.path.exists(f_albedo) or not os.path.exists(f_normal):
            print(f"\n[Skip] Missing Albedo/Normal for: {os.path.basename(f_beauty)}")
            continue

        try:
            # 3. Load Bitmaps
            bmp_beauty = mi.Bitmap(f_beauty)
            bmp_albedo = mi.Bitmap(f_albedo)
            bmp_normal = mi.Bitmap(f_normal)

            # 4. Check Resolution & Init Denoiser
            # We convert to a tuple (width, height) for safe comparison
            size_vec = bmp_beauty.size()
            size_tuple = (size_vec.x, size_vec.y)

            if denoiser is None or size_tuple != current_size:
                # Initialize OptiX Denoiser
                denoiser = mi.OptixDenoiser(size_vec, albedo=True, normals=True)
                current_size = size_tuple

            # 5. Convert to Tensors
            # The files are already strictly 3-channel RGB, so we can cast directly
            t_beauty = mi.TensorXf(bmp_beauty)
            t_albedo = mi.TensorXf(bmp_albedo)
            t_normal = mi.TensorXf(bmp_normal)

            # 6. RUN DENOISER
            # This is the magic step
            clean_tensor = denoiser(t_beauty, albedo=t_albedo, normals=t_normal)

            # 7. Save Result
            # Save as _clean.exr so you have a perfect target for training
            f_clean = f_beauty.replace("_beauty.exr", "_clean.exr")
            mi.util.write_bitmap(f_clean, clean_tensor)

        except Exception as e:
            print(f"\n[Error] Failed on {os.path.basename(f_beauty)}: {e}")

    print("\n\nPolishing Complete!")


if __name__ == "__main__":
    polish()
