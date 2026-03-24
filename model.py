import torch
import torch.nn as nn
import numpy as np

class HashEmbedder(nn.Module):
    def __init__(self, num_levels=12, base_res=16, max_res=512, log2_hashmap_size=15):
        super().__init__()
        self.num_levels = num_levels
        self.hashmap_size = 2**log2_hashmap_size
        b = np.exp((np.log(max_res) - np.log(base_res)) / (num_levels - 1))
        self.resolutions = [int(base_res * b**i) for i in range(num_levels)]

        self.embeddings = nn.ParameterList(
            [
                nn.Parameter(
                    torch.FloatTensor(self.hashmap_size, 2).uniform_(-1e-4, 1e-4)
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
            primes = [1, 2654435761, 805459861]
            p = x0 * torch.tensor(primes, device=x.device)
            h = (p[:, 0] ^ p[:, 1] ^ p[:, 2]) % self.hashmap_size
            outputs.append(embed[h])
        return torch.cat(outputs, dim=-1)


class HashNRC(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedder = HashEmbedder()
        self.net = nn.Sequential(
            nn.Linear(27, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 3),
            nn.Softplus(),
        )

    def forward(self, x_pos, x_norm):
        x_pos_norm = (x_pos + 2.0) / 4.0  # Normalize world space to [0,1]
        x_pos_norm = torch.clamp(x_pos_norm, 0.0, 1.0)
        embed = self.embedder(x_pos_norm)
        return self.net(torch.cat([embed, x_norm], dim=-1))
