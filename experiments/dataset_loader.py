import glob
import os
import random

import numpy as np
import torch


class SubsetPixelDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, subset_fraction=0.25):
        """
        Args:
            subset_fraction: 0.25 means "Load random 25% of the data"
        """
        self.scene_scale = 150

        # 1. Pick Random Files
        all_files = sorted(glob.glob(os.path.join(root_dir, "*.pt")))
        num_to_load = int(len(all_files) * subset_fraction)

        # Randomly sample files for this epoch
        self.files = random.sample(all_files, max(1, num_to_load))

        print(
            f">>> Loading {len(self.files)} files ({int(subset_fraction * 100)}% subset) into RAM..."
        )

        # 2. Load into RAM (Fast Access)
        all_pos = []
        all_albedo = []
        all_normal = []
        all_target = []

        for f_path in self.files:
            try:
                data = torch.load(f_path, map_location="cpu")

                # Extract & Sanitize
                pos_raw = torch.nan_to_num(data["pos"], nan=0.0)
                cam_pos = torch.nan_to_num(data["cam_pos"], nan=0.0)
                beauty = torch.nan_to_num(data["beauty"], nan=0.0)
                albedo = torch.nan_to_num(data["albedo"], nan=0.0)
                normal = torch.nan_to_num(data["normal"], nan=0.0)

                # Process
                pos_relative = pos_raw - cam_pos
                dist = torch.norm(pos_relative, dim=1, keepdim=True)
                mask = (dist < (self.scene_scale * 1.5)).float()
                pos_final = (pos_relative * mask) / self.scene_scale

                is_emitter = (beauty.mean(dim=-1, keepdim=True) > 1.0) & (
                    albedo.mean(dim=-1, keepdim=True) < 0.1
                )
                albedo = torch.where(is_emitter, torch.ones_like(albedo), albedo)

                epsilon = 0.01
                target_irradiance = beauty / (albedo + epsilon)
                target_final = torch.log1p(torch.clamp(target_irradiance, 0, 100.0))

                # Store as Float16 to save RAM (Crucial for 8GB Laptop)
                # We cast to Float32 during training
                all_pos.append(pos_final.permute(0, 2, 3, 1).reshape(-1, 3).half())
                all_albedo.append(albedo.permute(0, 2, 3, 1).reshape(-1, 3).half())
                all_normal.append(normal.permute(0, 2, 3, 1).reshape(-1, 3).half())
                all_target.append(
                    target_final.permute(0, 2, 3, 1).reshape(-1, 3).half()
                )

            except Exception:
                continue

        # 3. Merge
        self.pos = torch.cat(all_pos)
        self.albedo = torch.cat(all_albedo)
        self.normal = torch.cat(all_normal)
        self.target = torch.cat(all_target)

        print(f">>> RAM Ready: {self.pos.shape[0]} pixels.")

    def __len__(self):
        return self.pos.shape[0]

    def __getitem__(self, idx):
        # Return tensors (Trainer will convert to float32)
        return {
            "pos": self.pos[idx],
            "albedo": self.albedo[idx],
            "normal": self.normal[idx],
            "target": self.target[idx],
        }
