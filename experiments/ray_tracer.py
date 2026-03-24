import os

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# --- 1. CONFIGURATION ---
FRAME_PATH = "dataset_indoor_safe/coffee"
FILE_IDX = 0

# Performance Tuning
BATCH_SIZE = 2048  # Higher = Better Training, Lower = More FPS
LR = 0.01  # Hash Grids like high learning rates
ADAPTIVE_STRENGTH = 5.0


# --- 2. DEVICE SETUP (Auto-Detect) ---
def get_device():
    if torch.cuda.is_available():
        return "cuda"
    try:
        import torch_directml

        if torch_directml.is_available():
            print(">>> Using DirectML (Intel/AMD iGPU)")
            return torch_directml.device()
    except ImportError:
        pass
    return "cpu"


DEVICE = get_device()
print(f"Running on: {DEVICE}")


# --- 3. THE "INSTANT" NEURAL MODEL (Hash Grid) ---
class HashEmbedder(nn.Module):
    def __init__(
        self,
        num_levels=12,
        base_res=16,
        max_res=1024,
        log2_hashmap_size=17,
        features_per_level=2,
    ):
        super().__init__()
        self.num_levels = num_levels
        self.hashmap_size = 2**log2_hashmap_size
        b = np.exp((np.log(max_res) - np.log(base_res)) / (num_levels - 1))
        self.resolutions = [int(base_res * b**i) for i in range(num_levels)]

        self.embeddings = nn.ParameterList(
            [
                nn.Parameter(
                    torch.FloatTensor(self.hashmap_size, features_per_level).uniform_(
                        -1e-4, 1e-4
                    )
                )
                for _ in range(num_levels)
            ]
        )

    def forward(self, x):
        outputs = []
        for i, res in enumerate(self.resolutions):
            embed = self.embeddings[i]
            scaled_x = x * res
            x0 = torch.floor(scaled_x).long()

            # Spatial Hash (Primes)
            primes = [1, 2654435761, 805459861]
            p = x0 * torch.tensor(primes, device=x.device)
            h = (p[:, 0] ^ p[:, 1] ^ p[:, 2]) % self.hashmap_size

            outputs.append(embed[h])
        return torch.cat(outputs, dim=-1)


class HashNRC(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedder = HashEmbedder(num_levels=12)
        input_dim = (12 * 2) + 3  # 24 grid features + 3 normal

        # Tiny MLP (Fast Inference)
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 3),
        )

    def forward(self, x):
        x_pos = x[:, :3]
        x_norm = x[:, 3:]
        embedded_pos = self.embedder(x_pos)
        return self.net(torch.cat([embedded_pos, x_norm], dim=-1))


# --- 4. ROBUST DATA LOADING ---
def load_frame_data(folder_path, idx):
    os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

    # 1. Load Files
    try:
        files = sorted([f for f in os.listdir(folder_path) if "_beauty.exr" in f])
        if not files:
            raise FileNotFoundError(f"No EXR files in {folder_path}")
        base_name = files[idx].replace("_beauty.exr", "")
        print(f"Loading Scene: {base_name}")
    except Exception as e:
        print(f"Error: {e}")
        return None

    def read_exr(suffix):
        path = os.path.join(folder_path, f"{base_name}_{suffix}.exr")
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise FileNotFoundError(f"Missing {path}")
        return img

    beauty = read_exr("beauty")
    albedo = read_exr("albedo")
    normal = read_exr("normal")
    pos = read_exr("pos")

    H, W, _ = beauty.shape

    # --- THE GLASS FIX (Universal Pass-Through) ---
    # Logic: If an object is pitch black (Albedo < 0.05), it's either:
    # 1. A Light Source (Emissive)
    # 2. Glass (Transparent)
    # 3. Black Plastic (Dark)
    # In ALL these cases, dividing by Albedo (0.0) crashes the math.
    # We force Albedo to White (1.0). This tells the Neural Net:
    # "Don't shade this. Just output the final color directly."

    is_complex_material = albedo.mean(axis=2) < 0.05
    albedo[is_complex_material] = [1.0, 1.0, 1.0]
    # -----------------------------------------------

    # Flatten for Training
    flat_beauty = beauty.reshape(-1, 3)
    flat_albedo = albedo.reshape(-1, 3)
    flat_normal = normal.reshape(-1, 3)
    flat_pos = pos.reshape(-1, 3)
    noise = torch.randn_like(beauty) * beauty  # Signal-dependent noise
    noisy_radiance = beauty + (noise * 0.5)

    # 2. The Network must learn to predict the CLEAN light from this NOISY mess.
    # This is "Denoising" (The real job of NRC).
    noisy_targets = noisy_radiance / (albedo + 0.01)
    noisy_targets = torch.log1p(torch.clamp(noisy_targets, 0, 100))

    return {
        "beauty": noisy_targets.float().to(DEVICE),
        "albedo": torch.from_numpy(flat_albedo).float().to(DEVICE),
        "normal": torch.from_numpy(flat_normal).float().to(DEVICE),
        "pos": torch.from_numpy(flat_pos).float().to(DEVICE),
        "shape": (H, W),
        # Keep 2D tensor for rendering
        "full_albedo_tensor": torch.from_numpy(albedo).float().to(DEVICE),
    }


# --- 5. THE MAIN LOOP ---
def run_realtime_renderer():
    print(f"--- STARTING REAL-TIME NEURAL RENDERER ---")
    print("Controls: [S] Screenshot | [Q] Quit")

    data = load_frame_data(FRAME_PATH, FILE_IDX)
    if data is None:
        return
    H, W = data["shape"]
    num_pixels = H * W

    # Normalize Pos [0, 1] for Hash Grid
    raw_pos = data["pos"]
    pos_min = raw_pos.min(dim=0)[0]
    pos_max = raw_pos.max(dim=0)[0]
    pos_norm = (raw_pos - pos_min) / (pos_max - pos_min + 1e-5)

    inputs = torch.cat([pos_norm, data["normal"]], dim=1)  # [N, 6]

    # Calculate Target: Log Irradiance
    beauty_safe = torch.clamp(data["beauty"], 0, 100.0)
    targets = beauty_safe / (data["albedo"] + 0.01)
    targets = torch.log1p(torch.clamp(targets, 0, 100))

    # Model
    model = HashNRC().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR, betas=(0.9, 0.99))

    # Adaptive Error Buffer
    pixel_error_buffer = torch.ones(num_pixels, device=DEVICE)

    frame_count = 0

    while True:
        # --- PHASE 1: TRAINING (Adaptive) ---
        # 1. Importance Sampling
        safe_error = torch.nan_to_num(pixel_error_buffer, nan=0.001, posinf=1.0)
        safe_error = torch.clamp(safe_error, min=0.0001)
        probs = safe_error / safe_error.sum()

        idx = torch.multinomial(probs, BATCH_SIZE, replacement=True)

        batch_in = inputs[idx]
        batch_target = targets[idx]

        # 2. Backprop
        optimizer.zero_grad()
        pred = model(batch_in)

        loss_raw = (pred - batch_target) ** 2
        loss_per_sample = loss_raw.mean(dim=1)

        # Update Error History
        pixel_error_buffer[idx] = (
            0.9 * pixel_error_buffer[idx] + 0.1 * loss_per_sample.detach()
        )

        # Weighted Loss
        loss_final = loss_per_sample.mean() * ADAPTIVE_STRENGTH
        loss_final.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # Safety Clip
        optimizer.step()

        # --- PHASE 2: RENDERING (The Composite) ---
        # We render every few frames to keep UI responsive on slower GPUs
        # On RTX, you can set this to % 1
        if frame_count % 2 == 0:
            with torch.no_grad():
                # A. Predict Irradiance (The "Light Only")
                all_pred_log = model(inputs)
                irradiance = torch.expm1(all_pred_log)

                # B. THE COMPOSITE FIX (Multiplication)
                # Final = Irradiance (Light) * Albedo (Texture)
                # This restores sharpness instantly.
                final_img_flat = irradiance * data["albedo"]

                # C. Tone Mapping (ACES Filmic)
                # Matches Unreal Engine / Blender look
                x = final_img_flat
                a, b, c, d, e = 2.51, 0.03, 2.43, 0.59, 0.14
                final_img_flat = (x * (a * x + b)) / (x * (c * x + d) + e)
                final_img_flat = torch.clamp(final_img_flat, 0, 1)

                # Reshape to Image
                img_np = final_img_flat.cpu().numpy().reshape(H, W, 3)
                img_np = np.power(img_np, 1.0 / 2.2)  # Gamma Correction

                # D. Debug Heatmap
                heatmap = safe_error.reshape(H, W).cpu().numpy()
                heatmap = heatmap / (heatmap.max() + 1e-5)
                heatmap_c = cv2.applyColorMap(
                    (heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET
                )

                # Stack: [Final Render] | [Debug Heatmap]
                # img_np is already BGR because OpenCV loaded it that way. Don't swap it.
                combined = np.hstack([img_np, heatmap_c / 255.0])
                cv2.imshow("Neural Renderer (Live)", combined)

        # --- CONTROLS ---
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("s"):
            print(f"Saving frame_{frame_count}...")
            # Save clean render (0-255 uint8)
            clean_out = (img_np * 255).clip(0, 255).astype(np.uint8)
            cv2.imwrite(f"render_final_{frame_count:04d}.png", clean_out)

        frame_count += 1

    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_realtime_renderer()
