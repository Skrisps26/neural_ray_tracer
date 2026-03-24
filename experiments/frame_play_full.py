import os

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import mitsuba as mi
import numpy as np

# --- CONFIGURATION ---
DATA_ROOT = "dataset_multi"
FPS = 15  # Playback speed

# --- 1. INDEXING THE DATASET ---
print(f"Scanning '{DATA_ROOT}' for scenes...")

all_frames = []

# Find all scene folders (scene_000, scene_001, ...)
scene_folders = sorted(
    [
        d
        for d in os.listdir(DATA_ROOT)
        if os.path.isdir(os.path.join(DATA_ROOT, d)) and d.startswith("scene_")
    ]
)

if not scene_folders:
    print(
        f"Error: No 'scene_XXX' folders found in {DATA_ROOT}. Run generate_multiscene.py first."
    )
    exit()

# Flatten all frames into a single playlist
for scene_name in scene_folders:
    scene_path = os.path.join(DATA_ROOT, scene_name)

    # Find all GT files to count frames
    # naming convention: frame_XXXX_gt.exr
    gt_files = sorted(
        [
            f
            for f in os.listdir(scene_path)
            if f.endswith("_gt.exr") and f.startswith("frame_")
        ]
    )

    for gt_file in gt_files:
        # Extract base name (e.g., "frame_0000")
        base_name = gt_file.replace("_gt.exr", "")

        all_frames.append(
            {"scene": scene_name, "base_name": base_name, "path": scene_path}
        )

print(f"Found {len(scene_folders)} scenes containing {len(all_frames)} total frames.")

# --- 2. IMAGE PROCESSING ---


def tonemap(img):
    """Simple Gamma Correction + Exposure for viewing HDR data."""
    # Scale by 2.0 (Exposure) and pow 1/2.2 (Gamma)
    return np.clip((img * 2.0) ** (1 / 2.2), 0, 1)


def load_frame_data(global_index):
    # Get frame info from our playlist
    meta = all_frames[global_index]

    path_gt = os.path.join(meta["path"], f"{meta['base_name']}_gt.exr")
    path_noisy = os.path.join(meta["path"], f"{meta['base_name']}_noisy.exr")

    # Load via Mitsuba (handles EXR reading)
    bmp_gt = mi.Bitmap(path_gt)
    bmp_noisy = mi.Bitmap(path_noisy)

    # Convert to Numpy
    d_gt = np.array(bmp_gt)
    d_noisy = np.array(bmp_noisy)

    # Extract Channels
    # RGB is usually channels 0, 1, 2
    rgb_gt = tonemap(d_gt[:, :, :3])
    rgb_noisy = tonemap(d_noisy[:, :, :3])

    # Albedo is usually channels 3, 4, 5 (based on "albedo:albedo" being first AOV)
    # If d_gt has enough channels, grab them. Otherwise black.
    if d_gt.shape[2] >= 6:
        albedo = d_gt[:, :, 3:6]
    else:
        albedo = np.zeros_like(rgb_gt)

    title_text = f"{meta['scene']} : {meta['base_name']}"
    return rgb_noisy, rgb_gt, albedo, title_text


# --- 3. ANIMATION PLAYER ---

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 6))
plt.subplots_adjust(top=0.85)  # Make room for title

# Load first frame to initialize
img_n, img_g, img_a, txt = load_frame_data(0)

# Setup Axes
im1 = ax1.imshow(img_n)
ax1.set_title("Noisy Input (1 SPP)")
ax1.axis("off")

im2 = ax2.imshow(img_g)
ax2.set_title("Ground Truth (Clean)")
ax2.axis("off")

im3 = ax3.imshow(img_a)
ax3.set_title("Albedo (Aux)")
ax3.axis("off")

# Global Title
main_title = fig.suptitle(txt, fontsize=16, fontweight="bold")


def update(frame_idx):
    # Loop the video if it reaches the end
    idx = frame_idx % len(all_frames)

    noisy, gt, alb, title_str = load_frame_data(idx)

    im1.set_data(noisy)
    im2.set_data(gt)
    im3.set_data(alb)
    main_title.set_text(title_str)

    return [im1, im2, im3, main_title]


print("Starting playback...")
ani = animation.FuncAnimation(
    fig,
    update,
    frames=len(all_frames),
    interval=1000 / FPS,
    blit=False,  # False is sometimes more stable for text updates
)

plt.show()
