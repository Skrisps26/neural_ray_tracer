import numpy as np
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, num_freqs=12, log_scale=True):  # <--- Increased Frequencies
        super().__init__()
        self.num_freqs = num_freqs
        self.funcs = [torch.sin, torch.cos]

        if log_scale:
            self.freq_bands = 2.0 ** torch.linspace(0.0, num_freqs - 1, num_freqs)
        else:
            self.freq_bands = torch.linspace(1, 2.0 ** (num_freqs - 1), num_freqs)

    def forward(self, x):
        # x shape: (Batch_Size, 3)
        out = [x]
        for freq in self.freq_bands:
            for func in self.funcs:
                # Calculate sin(x * freq) and cos(x * freq)
                out.append(func(x * freq))

        # Concatenate everything
        return torch.cat(out, dim=-1)


class NeuralRenderer(nn.Module):
    def __init__(self, input_dim=81, hidden_dim=64):
        super().__init__()

        # 1. Positional Encoding
        # 12 freqs * 2 funcs * 3 coords = 72 dims
        # + 3 original coords = 75 dims
        self.pos_encoder = PositionalEncoding(num_freqs=12)

        # 2. Input Dimension Calculation
        # 75 (Encoded Pos) + 3 (Albedo) + 3 (Normal) = 81
        self.input_dim = input_dim

        # 3. The MLP (Tiny Brain)
        # We use a standard Residual MLP for speed and stability
        self.layers = nn.Sequential(
            nn.Linear(self.input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 3),  # Output: RGB Irradiance
        )

        # Initialize weights for stability (helps prevent NaN start)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, albedo, normal, pos):
        """
        Args:
            albedo: (N, 3)  <-- Already flattened!
            normal: (N, 3)
            pos:    (N, 3)
        """

        # --- THE FIX IS HERE ---
        # Old Code:
        # pos = pos.permute(0, 2, 3, 1).reshape(-1, 3)  <-- DELETED
        # albedo = albedo.permute(0, 2, 3, 1).reshape(-1, 3) <-- DELETED
        # normal = normal.permute(0, 2, 3, 1).reshape(-1, 3) <-- DELETED

        # 1. Encode Position
        # (N, 3) -> (N, 75)
        pos_encoded = self.pos_encoder(pos)

        # 2. Concatenate Features
        # (N, 75) + (N, 3) + (N, 3) -> (N, 81)
        model_input = torch.cat([pos_encoded, albedo, normal], dim=-1)

        # 3. Inference
        # (N, 81) -> (N, 3)
        output = self.layers(model_input)

        return output
