import os

import cv2
import numpy as np
import torch

# --- CONFIG ---
FRAME_PATH = "dataset_indoor_safe/coffee"
FILE_IDX = 0
# ----------------


def robust_render():
    os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

    # 1. Load Data (Robustly)
    try:
        files = sorted([f for f in os.listdir(FRAME_PATH) if "_beauty.exr" in f])
        if not files:
            raise FileNotFoundError("No EXR files found")
        base_name = files[FILE_IDX].replace("_beauty.exr", "")
    except Exception as e:
        print(f"Error: {e}")
        return

    def read_exr(suffix):
        f = os.path.join(FRAME_PATH, f"{base_name}_{suffix}.exr")
        return cv2.imread(f, cv2.IMREAD_UNCHANGED)

    beauty = read_exr("beauty")
    albedo = read_exr("albedo")

    # 2. THE FIX FOR BLACK LIGHTS
    # We create a "Corrected Albedo" mask.
    # Logic: If the Albedo is Pitch Black (< 0.05) ...
    # ... AND the ground truth is Bright (> 0.5), it MUST be a light.
    is_emitter = (albedo.mean(axis=2) < 0.05) & (beauty.mean(axis=2) > 0.5)

    # Force Albedo to White (1.0) where we detect lights
    albedo_corrected = albedo.copy()
    albedo_corrected[is_emitter] = [1.0, 1.0, 1.0]

    # 3. SIMULATE THE NEURAL NETWORK (Ideal Case)
    # Since we are debugging the COMPOSITE, we will simulate a "Perfect" Neural Net.
    # A perfect NRC predicts: Beauty / Corrected_Albedo
    ideal_irradiance = beauty / (albedo_corrected + 0.001)

    # We BLUR it to simulate what the Real Neural Net does (it's never perfectly sharp)
    nrc_simulation = cv2.GaussianBlur(ideal_irradiance, (15, 15), 0)

    # 4. THE COMPOSITE (Clean Math Only)
    # Formula: Final = Irradiance (Blurry) * Albedo (Sharp)
    # This is how game engines do it. No halo hacks.
    final_render = nrc_simulation * albedo_corrected

    # 5. Tone Mapping (Standard ACES approximation)
    final_render = final_render / (final_render + 1.0)
    final_render = np.power(final_render, 1.0 / 2.2)

    # Display
    cv2.imshow("1. Corrected Albedo (Check Lights)", albedo_corrected)
    cv2.imshow("2. Simulated NRC (Blurry Light)", nrc_simulation)
    cv2.imshow("3. Final Composite (Sharp)", final_render)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    robust_render()
