import os

import imageio
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import mitsuba as mi
import numpy as np

# --- CONFIGURATION ---
DATASET_ROOT = "dataset_indoor_safe"
OUTPUT_VIDEO = "dataset_preview.mp4"
FPS = 24
FIG_SIZE = (12, 6)  # Window size

# --- 1. INDEX THE DATASET ---
print(f"Scanning '{DATASET_ROOT}'...")

all_frames = []
scene_folders = sorted([f.path for f in os.scandir(DATASET_ROOT) if f.is_dir()])

if not scene_folders:
    print(f"Error: No scene folders found in {DATASET_ROOT}")
    exit()

for scene_path in scene_folders:
    scene_name = os.path.basename(scene_path)

    # We look for the noisy input first
    noisy_files = sorted(
        [f for f in os.listdir(scene_path) if f.endswith("_noisy.exr")]
    )

    for noisy_file in noisy_files:
        # Extract frame number prefix (e.g., "frame_0001")
        base_name = noisy_file.replace("_noisy.exr", "")

        # Determine the best Ground Truth available
        # Priority: Clean (Denoised) > Beauty (Raw High-SPP)
        path_clean = os.path.join(scene_path, f"{base_name}_clean.exr")
        path_beauty = os.path.join(scene_path, f"{base_name}_beauty.exr")

        target_path = None
        target_type = "Unknown"

        if os.path.exists(path_clean):
            target_path = path_clean
            target_type = "Clean (Denoised)"
        elif os.path.exists(path_beauty):
            target_path = path_beauty
            target_type = "Beauty (Raw 512spp)"

        if target_path:
            all_frames.append(
                {
                    "scene": scene_name,
                    "frame": base_name,
                    "path_noisy": os.path.join(scene_path, noisy_file),
                    "path_target": target_path,
                    "type": target_type,
                }
            )

print(f"Found {len(all_frames)} valid frames.")
if len(all_frames) == 0:
    print("No complete frame pairs found. Did you run the generator?")
    exit()


# --- 2. IMAGE UTILS ---
def tonemap(img):
    """
    Standard Gamma 2.2 correction for viewing HDR EXR files on a screen.
    Without this, images look extremely dark.
    """
    # Simple Reinhard-ish tonemap: x / (x + 1) to handle bright spots?
    # Or just simple Gamma. Let's stick to Gamma 2.2 for training data visualization.
    return np.clip(np.power(img, 1 / 2.2), 0, 1)


def load_frame_pair(idx):
    meta = all_frames[idx]

    # Load High Dynamic Range images
    bmp_noisy = mi.Bitmap(meta["path_noisy"])
    bmp_target = mi.Bitmap(meta["path_target"])

    # Convert to NumPy and discard Alpha channel if present
    img_noisy = np.array(bmp_noisy)[:, :, :3]
    img_target = np.array(bmp_target)[:, :, :3]

    # Tonemap for display
    img_noisy = tonemap(img_noisy)
    img_target = tonemap(img_target)

    title = f"{meta['scene']} : {meta['frame']} | {meta['type']}"
    return img_noisy, img_target, title


# --- 3. VIDEO RENDERER ---
print(f"Rendering video to {OUTPUT_VIDEO}...")
writer = imageio.get_writer(OUTPUT_VIDEO, fps=FPS)

for i in range(len(all_frames)):
    print(f"Processing frame {i + 1}/{len(all_frames)}...", end="\r")

    noisy, target, _ = load_frame_pair(i)

    # Stitch side-by-side
    combined = np.hstack((noisy, target))

    # Convert float (0.0 - 1.0) to uint8 (0 - 255) for video
    combined_u8 = (combined * 255).astype(np.uint8)
    writer.append_data(combined_u8)

writer.close()
print("\nVideo Saved!")

# --- 4. LIVE PLAYER ---
print("Launching Player...")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=FIG_SIZE)
plt.subplots_adjust(top=0.90, bottom=0.05, left=0.05, right=0.95)

# Initial Frame
n, t, title_str = load_frame_pair(0)
im1 = ax1.imshow(n)
ax1.set_title("Input (Noisy 2spp)")
ax1.axis("off")

im2 = ax2.imshow(t)
ax2.set_title("Target (Ground Truth)")
ax2.axis("off")

sup_title = fig.suptitle(title_str, fontsize=14, fontweight="bold")


def update(frame_idx):
    n, t, title_str = load_frame_pair(frame_idx)
    im1.set_data(n)
    im2.set_data(t)
    sup_title.set_text(title_str)
    return im1, im2, sup_title


ani = animation.FuncAnimation(
    fig, update, frames=len(all_frames), interval=1000 / FPS, blit=False
)

plt.show()
