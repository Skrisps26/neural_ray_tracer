import os
import random

import drjit as dr
import mitsuba as mi
import numpy as np

# Hardware check
try:
    mi.set_variant("cuda_ad_rgb")
except:
    mi.set_variant("scalar_rgb")

OUTPUT_ROOT = "dataset_multi"
os.makedirs(OUTPUT_ROOT, exist_ok=True)

# --- CONFIGURATION ---
NUM_SCENES = 10  # How many unique worlds to build
FRAMES_PER_SCENE = 30  # How long to film each world
RES_X, RES_Y = 512, 512
SPP_GT = 512  # High quality target
SPP_NOISY = 2  # Noisy input

# --- ASSET LIBRARY ---
# We randomly pick from these to build the world


def get_random_material():
    """Returns a random material dictionary."""
    mats = [
        # 1. Gold (Mirror-like)
        {"type": "roughconductor", "material": "Au", "alpha": 0.05},
        # 2. Silver (Bright Mirror)
        {"type": "roughconductor", "material": "Ag", "alpha": 0.05},
        # 3. Glass (Frosted)
        {"type": "roughdielectric", "int_ior": 1.5, "alpha": 0.05},
        # 4. Matte Red
        {"type": "diffuse", "reflectance": {"type": "rgb", "value": [0.8, 0.1, 0.1]}},
        # 5. Matte Blue
        {"type": "diffuse", "reflectance": {"type": "rgb", "value": [0.1, 0.1, 0.8]}},
        # 6. Matte White (Floor-like)
        {"type": "diffuse", "reflectance": {"type": "rgb", "value": [0.8, 0.8, 0.8]}},
    ]
    return random.choice(mats)


def get_random_transform(z_offset=0):
    """Returns a random scale/rotation/position transform."""
    x = random.uniform(-2.0, 2.0)
    y = random.uniform(-2.0, 2.0)
    scale = random.uniform(0.3, 0.8)
    rot = random.uniform(0, 360)

    return (
        mi.ScalarTransform4f.translate([x, y, scale + z_offset])
        .rotate([0, 0, 1], rot)
        .scale(scale)
    )


# --- SCENE BUILDER ---


def generate_scene_layout(scene_seed):
    """
    Procedurally generates a unique scene dictionary based on a seed.
    """
    random.seed(scene_seed)
    np.random.seed(scene_seed)

    # Base Scene Dict
    scene = {
        "type": "scene",
        "integrator": {
            "type": "aov",
            "aovs": "albedo:albedo, nn:sh_normal, pos:position",
            "sample_integrator": {"type": "path", "max_depth": 8},
        },
        "sensor": {
            "type": "perspective",
            "fov": 45,
            # Placeholder transform, will be updated per frame
            "to_world": mi.ScalarTransform4f.look_at([0, -6, 3], [0, 0, 0], [0, 0, 1]),
            "film": {
                "type": "hdrfilm",
                "width": RES_X,
                "height": RES_Y,
                "pixel_format": "rgb",
                "rfilter": {"type": "box"},
            },
        },
        # Floor (Standard)
        "floor": {
            "type": "rectangle",
            "to_world": mi.ScalarTransform4f.scale(10),
            "bsdf": {
                "type": "diffuse",
                "reflectance": {"type": "rgb", "value": [0.5, 0.5, 0.5]},
            },
        },
        # Fill Light (Always present)
        "light_fill": {
            "type": "constant",
            "radiance": {"type": "rgb", "value": [0.05, 0.05, 0.06]},
        },
    }

    # 1. Randomize Key Light Position
    light_x = random.uniform(-3, 3)
    light_y = random.uniform(-3, 3)
    scene["light_key"] = {
        "type": "point",
        "position": [light_x, light_y, 6],
        "intensity": {"type": "rgb", "value": [300, 300, 280]},
    }

    # 2. Scatter 3-6 Random Objects
    num_objects = random.randint(3, 6)
    for i in range(num_objects):
        obj_type = random.choice(["cube", "sphere"])

        # Ensure they don't spawn inside floor (z_offset)
        z_off = 0.0 if obj_type == "cube" else 0.0  # Pivot is center

        obj_dict = {
            "type": obj_type,
            "to_world": get_random_transform(z_off),
            "bsdf": get_random_material(),
        }
        scene[f"obj_{i}"] = obj_dict

    return scene


# --- MAIN LOOP ---

print(f"Starting Production: {NUM_SCENES} Scenes, {FRAMES_PER_SCENE} Frames each.")
print(f"Total Frames: {NUM_SCENES * FRAMES_PER_SCENE}")

for s_idx in range(NUM_SCENES):
    print(f"\n--- Generatng Scene {s_idx:03d} ---")

    # 1. Create the fixed layout for this scene
    # We use the index as the seed so the objects don't move between frames!
    scene_dict = generate_scene_layout(scene_seed=s_idx)

    # 2. Setup Folder
    scene_dir = os.path.join(OUTPUT_ROOT, f"scene_{s_idx:03d}")
    os.makedirs(scene_dir, exist_ok=True)

    # 3. Render Loop (Camera Animation Only)
    for f_idx in range(FRAMES_PER_SCENE):
        print(f"Rendering Frame {f_idx + 1}/{FRAMES_PER_SCENE}...", end="\r")

        # Calculate Camera Move (Orbit)
        t = f_idx / FRAMES_PER_SCENE
        angle = t * 2 * np.pi
        cam_x = np.sin(angle * 0.5) * 6.0
        cam_y = -np.cos(angle * 0.5) * 6.0
        cam_z = 3.0 + np.sin(t * 4) * 0.5  # Add slight bob

        # Update Camera in Dict
        scene_dict["sensor"]["to_world"] = mi.ScalarTransform4f.look_at(
            origin=[cam_x, cam_y, cam_z], target=[0, 0, 0.5], up=[0, 0, 1]
        )

        # Load & Render
        scene = mi.load_dict(scene_dict)

        # GT Pass
        img_gt = mi.render(scene, spp=SPP_GT)

        # Noisy Pass
        img_noisy = mi.render(scene, spp=SPP_NOISY)

        # Save
        frame_name = f"frame_{f_idx:04d}"
        mi.util.write_bitmap(f"{scene_dir}/{frame_name}_gt.exr", img_gt)
        mi.util.write_bitmap(f"{scene_dir}/{frame_name}_noisy.exr", img_noisy)

        # Save Cam Matrix
        sensor = scene.sensors()[0]
        cam_to_world = sensor.world_transform().matrix.numpy()
        np.save(f"{scene_dir}/{frame_name}_cam.npy", cam_to_world)

print("\n\nMass Production Complete!")
