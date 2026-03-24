import torch

# --- CONFIG ---
RES_X, RES_Y = 640, 480
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LOOKAHEAD_FRAMES = 15.0  # How far into the future do we peek?
FUTURE_BUDGET = 0.30     # Proportion of resources for future state training
