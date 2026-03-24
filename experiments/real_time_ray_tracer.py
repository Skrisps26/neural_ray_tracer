import os

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# --- 1. CONFIG & DEVICE SETUP ---
FRAME_PATH = "dataset_indoor_safe/coffee"
FILE_IDX = 0

# Game Constraints (Adjust for your GPU)
BATCH_SIZE = 1024  # Safe for iGPUs (Increase to 4096 for RTX cards)
ADAPTIVE_STRENGTH = 5.0
LR = 0.01


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    try:
        import torch_directml

        if torch_directml.is_available():
            print("Using DirectML (iGPU/AMD/Intel)")
            return torch_directml.device()
    except ImportError:
        pass
    return "cpu"


DEVICE = get_device()
print(f"Running on: {DEVICE}")


# --- 2. HASH GRID MODEL (The Instant-NeRF Tech) ---
class HashEmbedder(nn.Module):
    def __init__(
        self,
        num_levels=8,
        base_res=16,
        max_res=512,
        log2_hashmap_size=15,
        features_per_level=2,
    ):
        super().__init__()
        self.num_levels = num_levels
        self.hashmap_size = 2**log2_hashmap_size

        # Calculate grid resolutions (Geometric Progression)
        b = np.exp((np.log(max_res) - np.log(base_res)) / (num_levels - 1))
        self.resolutions = [int(base_res * b**i) for i in range(num_levels)]

        # Learnable Feature Grid
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
        # x is [N, 3] in range [0, 1]
        outputs = []
        for i, res in enumerate(self.resolutions):
            embed = self.embeddings[i]
            scaled_x = x * res

            # 1. Hashing
            x0 = torch.floor(scaled_x).long()

            # Simple Spatial Hash (Primes to scramble coords)
            primes = [1, 2654435761, 805459861]
            p = x0 * torch.tensor(primes, device=x.device)
            h = (p[:, 0] ^ p[:, 1] ^ p[:, 2]) % self.hashmap_size

            # 2. Lookup (Nearest Neighbor for Speed)
            outputs.append(embed[h])

        return torch.cat(outputs, dim=-1)


class HashNRC(nn.Module):
    def __init__(self):
        super().__init__()
        # 8 levels * 2 features = 16 inputs from grid + 3 normals = 19 total
        self.embedder = HashEmbedder(num_levels=8)
        input_dim = (8 * 2) + 3

        # Tiny MLP
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 3),
        )

    def forward(self, x):
        # FIX: Split the input tensor [N, 6] -> Pos [N, 3] + Normal [N, 3]
        x_pos = x[:, :3]
        x_norm = x[:, 3:]

        # Embed Position
        embedded_pos = self.embedder(x_pos)

        # Concatenate & Predict
        features = torch.cat([embedded_pos, x_norm], dim=-1)
        return self.net(features)


# --- 3. ROBUST DATA LOADER ---
def load_frame_data(folder_path, idx):
    # Auto-find the correct file prefix
    try:
        files = sorted([f for f in os.listdir(folder_path) if "_beauty.exr" in f])
        if len(files) <= idx:
            raise FileNotFoundError("Frame index out of range")
        base_name = files[idx].replace("_beauty.exr", "")
        print(f"Loading: {base_name}")
    except Exception as e:
        print(f"Error finding files: {e}")
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

    # Flatten
    beauty_flat = beauty.reshape(-1, 3)
    albedo_flat = albedo.reshape(-1, 3)
    normal_flat = normal.reshape(-1, 3)
    pos_flat = pos.reshape(-1, 3)

    # Fix "Black Albedo" on Lights
    is_light = (beauty_flat.mean(axis=1) > 1.0) & (albedo_flat.mean(axis=1) < 0.1)
    albedo_flat[is_light] = [1.0, 1.0, 1.0]

    return {
        "beauty": torch.from_numpy(beauty_flat).float().to(DEVICE),
        "albedo": torch.from_numpy(albedo_flat).float().to(DEVICE),
        "normal": torch.from_numpy(normal_flat).float().to(DEVICE),
        "pos": torch.from_numpy(pos_flat).float().to(DEVICE),
        "shape": (H, W),
    }


# --- 4. THE MAIN LOOP ---
def run_adaptive_simulation():
    print(f"--- STARTING REAL-TIME NEURAL RADIANCE CACHE ---")

    data = load_frame_data(FRAME_PATH, FILE_IDX)
    if data is None:
        return
    H, W = data["shape"]
    num_pixels = H * W

    # Normalize Position to strictly [0, 1] for Hash Grid
    raw_pos = data["pos"]
    pos_min = raw_pos.min(dim=0)[0]
    pos_max = raw_pos.max(dim=0)[0]
    pos_norm = (raw_pos - pos_min) / (pos_max - pos_min + 1e-5)  # Avoid div/0

    inputs = torch.cat([pos_norm, data["normal"]], dim=1)  # [N, 6]

    # Calculate Targets (Log Irradiance)
    beauty_safe = torch.clamp(data["beauty"], 0, 100.0)
    targets = beauty_safe / (data["albedo"] + 0.01)
    targets = torch.log1p(torch.clamp(targets, 0, 100))

    # Init Model & Optimizer
    model = HashNRC().to(DEVICE)
    # Hash Grids need Adam, not SGD
    optimizer = optim.Adam(model.parameters(), lr=LR, betas=(0.9, 0.99))

    # Error Buffer for Adaptive Sampling
    pixel_error_buffer = torch.ones(num_pixels, device=DEVICE)

    print("Controls: 'Q' to Quit, 'S' to Save Screenshot")

    frame_count = 0

    while True:
        # --- 1. ADAPTIVE SAMPLING ---
        # Safe prob calc
        safe_error = torch.nan_to_num(pixel_error_buffer, nan=0.001, posinf=1.0)
        safe_error = torch.clamp(safe_error, min=0.0001)
        probs = safe_error / safe_error.sum()

        idx = torch.multinomial(probs, BATCH_SIZE, replacement=True)

        batch_in = inputs[idx]
        batch_target = targets[idx]

        # --- 2. TRAINING ---
        optimizer.zero_grad()
        pred = model(batch_in)  # Now works because forward accepts 1 arg

        loss_raw = (pred - batch_target) ** 2
        loss_per_sample = loss_raw.mean(dim=1)

        # Update Error Buffer (Exponential Moving Average)
        pixel_error_buffer[idx] = (
            0.9 * pixel_error_buffer[idx] + 0.1 * loss_per_sample.detach()
        )

        # Weighted Loss
        loss_final = loss_per_sample.mean() * ADAPTIVE_STRENGTH
        loss_final.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        # --- 3. RENDERING (Interleaved for Speed) ---
        # Render every frame, but maybe skip complex stuff on slow GPUs if needed
        if frame_count % 1 == 0:
            with torch.no_grad():
                # Predict Light
                all_pred_log = model(inputs)
                nrc_irradiance = torch.expm1(all_pred_log)

                # TEXTURE MULTIPLICATION (The Fix for Blur)
                # Final = Irradiance * Albedo
                final_img = nrc_irradiance * data["albedo"]

                # Tone Map
                img_np = final_img.cpu().numpy().reshape(H, W, 3)
                img_np = img_np / (img_np + 1.0)
                img_np = np.power(img_np, 1.0 / 2.2)

                # Heatmap
                heatmap = safe_error.reshape(H, W).cpu().numpy()
                heatmap = heatmap / (heatmap.max() + 1e-5)
                heatmap_color = cv2.applyColorMap(
                    (heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET
                )

                # Combine
                combined = np.hstack(
                    [cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR), heatmap_color / 255.0]
                )
                cv2.imshow("Real-Time Neural Ray Tracing", combined)

        # --- 4. CONTROLS ---
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("s"):
            print(f"Saving frame_{frame_count:04d}...")
            clean_out = (img_np * 255).clip(0, 255).astype(np.uint8)
            cv2.imwrite(
                f"render_{frame_count:04d}.png",
                cv2.cvtColor(clean_out, cv2.COLOR_RGB2BGR),
            )

        frame_count += 1

    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_adaptive_simulation()
