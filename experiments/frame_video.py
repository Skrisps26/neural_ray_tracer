import os

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import mitsuba as mi
import numpy as np

DATA_DIR = "dataset_output"
frames = sorted([f for f in os.listdir(DATA_DIR) if f.startswith("frame_")])


def tonemap(img):
    # Simple exposure + Gamma correction
    return np.clip((img * 2.0) ** (1 / 2.2), 0, 1)


def load_frame(idx):
    path = os.path.join(DATA_DIR, frames[idx])

    # Load EXRs
    # Slice [:,:,0:3] to take just the RGB Beauty pass
    # Slice [:,:,3:6] to take Albedo (check your AOV string order!)
    # In the generator "aovs": "albedo:albedo, ..." -> Albedo is usually ch 4,5,6

    bmp_noisy = mi.Bitmap(f"{path}/noisy.exr")
    bmp_gt = mi.Bitmap(f"{path}/gt.exr")

    d_noisy = np.array(bmp_noisy)
    d_gt = np.array(bmp_gt)

    # Beauty is usually first 3 channels
    rgb_noisy = tonemap(d_noisy[:, :, :3])
    rgb_gt = tonemap(d_gt[:, :, :3])

    # Albedo is usually channels 3,4,5 (if alpha is present/absent order varies)
    # We'll just grab channels 3:6
    albedo = d_gt[:, :, 3:6]

    return rgb_noisy, rgb_gt, albedo


fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
n, g, a = load_frame(0)
im1 = ax1.imshow(n)
ax1.set_title("Noisy Input")
im2 = ax2.imshow(g)
ax2.set_title("Clean GT")
im3 = ax3.imshow(a)
ax3.set_title("Albedo AOV")


def update(i):
    n, g, a = load_frame(i % len(frames))
    im1.set_data(n)
    im2.set_data(g)
    im3.set_data(a)


ani = animation.FuncAnimation(fig, update, frames=len(frames), interval=50)
plt.show()
