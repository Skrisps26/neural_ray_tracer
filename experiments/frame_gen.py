import os

import drjit as dr
import mitsuba as mi
import numpy as np

# Try GPU, fall back to CPU
try:
    mi.set_variant("cuda_ad_rgb")
except:
    mi.set_variant("scalar_rgb")

OUTPUT_DIR = "dataset_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# CONFIGURATION
NUM_FRAMES = 50
RES_X, RES_Y = 512, 512
SPP_GT = 512  # Clean target
SPP_NOISY = 2  # Noisy input


def create_scene_dict(frame_idx, total_frames):
    t = frame_idx / total_frames
    angle = t * 2 * np.pi

    # 1. Camera Orbit (Radius 6.0)
    cam_x = np.sin(angle * 0.5) * 6.0
    cam_y = -np.cos(angle * 0.5) * 6.0
    cam_z = 3.0

    cam_transform = mi.ScalarTransform4f.look_at(
        origin=[cam_x, cam_y, cam_z], target=[0, 0, 0.5], up=[0, 0, 1]
    )

    # 2. Objects
    # Rotating Gold Cube
    cube_transform = (
        mi.ScalarTransform4f.translate([0, 0, 0.5])
        .rotate([0, 0, 1], angle * 180 / np.pi)
        .scale(0.6)
    )

    # Bobbing Glass Sphere
    sphere_transform = mi.ScalarTransform4f.translate(
        [-1.2, 0, 1.0 + 0.3 * np.sin(angle * 3)]
    ).scale(0.4)

    return {
        "type": "scene",
        "integrator": {
            "type": "aov",
            # We extract Albedo, Normal, and Position for your G-Buffer
            "aovs": "albedo:albedo, nn:sh_normal, pos:position",
            "sample_integrator": {
                "type": "path",
                "max_depth": 8,
            },
        },
        "sensor": {
            "type": "perspective",
            "fov": 45,
            "to_world": cam_transform,
            "film": {
                "type": "hdrfilm",
                "width": RES_X,
                "height": RES_Y,
                "pixel_format": "rgb",
                "rfilter": {"type": "box"},
            },
        },
        # --- LIGHTING FIX ---
        # 1. Main Key Light (Point Light)
        # Positioned offset [2, 2, 6] so it creates nice shadows
        "light_key": {
            "type": "point",
            "position": [2, 2, 6],
            "intensity": {
                "type": "rgb",
                "value": [300, 300, 280],  # High intensity to reach floor
            },
        },
        # 2. Fill Light (Environment)
        # Ensures shadows are not pitch black (0.0)
        "light_fill": {
            "type": "constant",
            "radiance": {"type": "rgb", "value": [0.05, 0.05, 0.06]},
        },
        # --- MATERIALS ---
        "floor": {
            "type": "rectangle",
            "to_world": mi.ScalarTransform4f.scale(10),
            "bsdf": {
                "type": "diffuse",
                "reflectance": {"type": "rgb", "value": [0.5, 0.5, 0.5]},
            },
        },
        "cube_gold": {
            "type": "cube",
            "to_world": cube_transform,
            "bsdf": {"type": "roughconductor", "material": "Au", "alpha": 0.05},
        },
        "hero_sphere_glass": {
            "type": "sphere",
            "to_world": sphere_transform,
            "bsdf": {
                "type": "roughdielectric",  # CHANGE THIS: from 'dielectric'
                "int_ior": 1.5,
                "ext_ior": 1.0,
                "alpha": 0.05,  # ADD THIS: 0.0 is perfect, 0.2 is very frosty
                "distribution": "ggx",
            },
        },
        "sphere_red": {
            "type": "sphere",
            "to_world": mi.ScalarTransform4f.translate([1.2, 0.5, 0.5]).scale(0.5),
            "bsdf": {
                "type": "diffuse",
                "reflectance": {"type": "rgb", "value": [0.8, 0.1, 0.1]},
            },
        },
    }


print(f"Generating {NUM_FRAMES} frames...")

for i in range(NUM_FRAMES):
    print(f"Frame {i + 1}/{NUM_FRAMES}", end=" ")

    scene = mi.load_dict(create_scene_dict(i, NUM_FRAMES))
    frame_dir = os.path.join(OUTPUT_DIR, f"frame_{i:04d}")
    os.makedirs(frame_dir, exist_ok=True)

    # Render GT
    img_gt = mi.render(scene, spp=SPP_GT)

    # SAFETY CHECK: Print max brightness
    # If this is 0.0, the light is broken.
    max_val = dr.max(img_gt)
    peak_b = max_val.numpy().item()
    print(f"| Peak Brightness: {peak_b:.2f}", end="")
    if peak_b == 0:
        print(" [ERROR: BLACK FRAME]")
    else:
        print(" [OK]")

    # Render Noisy
    img_noisy = mi.render(scene, spp=SPP_NOISY)

    # Save RAW Tensors (Fixes your "RuntimeError")
    # We do NOT convert to bitmap first. We write the tensor directly.
    mi.util.write_bitmap(f"{frame_dir}/gt.exr", img_gt)
    mi.util.write_bitmap(f"{frame_dir}/noisy.exr", img_noisy)

    # Save Camera Matrix
    sensor = scene.sensors()[0]
    cam_to_world = sensor.world_transform().matrix.numpy()
    np.save(f"{frame_dir}/cam_to_world.npy", cam_to_world)

print("\nDone!")
