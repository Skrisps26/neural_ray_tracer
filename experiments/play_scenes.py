import os

import imageio
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import mitsuba as mi
import numpy as np

# --- CONFIGURATION ---
DATASET_ROOT = (
    "dataset_indoor_realism"  # Folder containing scene_LivingRoom, scene_Bedroom, etc.
)
OUTPUT_VIDEO = "dataset_preview.mp4"
FPS = 24
FIG_SIZE = (12, 6)  # Width, Height in inches

# --- 1. INDEX THE DATASET ---
print(f"Scanning '{DATASET_ROOT}'...")

all_frames = []
scene_folders = sorted([f.path for f in os.scandir(DATASET_ROOT) if f.is_dir()])

if not scene_folders:
    print(f"Error: No scene folders found in {DATASET_ROOT}")
    exit()

print(f"Found {len(scene_folders)} scenes.")

for scene_path in scene_folders:
    scene_name = os.path.basename(scene_path)

    # Find all GT frames
    gt_files = sorted([f for f in os.listdir(scene_path) if f.endswith("_gt.exr")])

    for gt_file in gt_files:
        # Construct pair paths
        base_name = gt_file.replace("_gt.exr", "")
        path_gt = os.path.join(scene_path, gt_file)
        path_noisy = os.path.join(scene_path, f"{base_name}_noisy.exr")

        if os.path.exists(path_noisy):
            all_frames.append(
                {
                    "scene": scene_name,
                    "frame": base_name,
                    "gt_path": path_gt,
                    "noisy_path": path_noisy,
                }
            )

print(f"Total Frames: {len(all_frames)}")
if len(all_frames) == 0:
    exit()


# --- 2. IMAGE PROCESSING UTILS ---
def tonemap(img):
    """Simple Exposure + Gamma for viewing HDR EXRs"""
    return np.clip((img * 2.5) ** (1 / 2.2), 0, 1)


def load_frame_pair(idx):
    meta = all_frames[idx]

    # Load via Mitsuba
    bmp_gt = mi.Bitmap(meta["gt_path"])
    bmp_noisy = mi.Bitmap(meta["noisy_path"])

    # Convert to Numpy & Tonemap
    # Slicing [:3] ensures we only get RGB, dropping Alpha/Depth if present
    img_gt = tonemap(np.array(bmp_gt)[:, :, :3])
    img_noisy = tonemap(np.array(bmp_noisy)[:, :, :3])

    return img_noisy, img_gt, f"{meta['scene']} : {meta['frame']}"


# --- 3. VIDEO RENDERER (TO DISK) ---
print(f"Rendering video to {OUTPUT_VIDEO}...")
writer = imageio.get_writer(OUTPUT_VIDEO, fps=FPS)


# We define a helper to stitch images side-by-side for the video
def make_frame_image(idx):
    noisy, gt, title = load_frame_pair(idx)

    # Stack horizontally: [Noisy | GT]
    combined = np.hstack((noisy, gt))

    # Convert from 0-1 float to 0-255 uint8 for video
    combined_u8 = (combined * 255).astype(np.uint8)
    return combined_u8


# Render Loop
for i in range(len(all_frames)):
    print(f"Processing frame {i + 1}/{len(all_frames)}...", end="\r")
    frame_data = make_frame_image(i)
    writer.append_data(frame_data)

writer.close()
print("\nVideo Saved Successfully!")

# --- 4. LIVE PLAYER (PLAYBACK) ---
print("Launching Player...")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=FIG_SIZE)
plt.subplots_adjust(top=0.90, bottom=0.05, left=0.05, right=0.95)

# Init first frame
n, g, t = load_frame_pair(0)
im1 = ax1.imshow(n)
ax1.set_title("Input (Noisy 1spp)")
ax1.axis("off")

im2 = ax2.imshow(g)
ax2.set_title("Target (GT 512spp)")
ax2.axis("off")

title_text = fig.suptitle(t, fontsize=14, fontweight="bold")


def update(frame_idx):
    n, g, t = load_frame_pair(frame_idx)
    im1.set_data(n)
    im2.set_data(g)
    title_text.set_text(t)
    return im1, im2, title_text


# Create Animation
ani = animation.FuncAnimation(
    fig, update, frames=len(all_frames), interval=1000 / FPS, blit=False, repeat=True
)

plt.show()
