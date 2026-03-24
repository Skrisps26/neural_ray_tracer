import glob
import os

import torch

# Get the first chunk file
file_path = sorted(glob.glob("dataset_optimized/*.pt"))[0]
data = torch.load(file_path)

print("Keys in chunk:", data.keys())

if "cam" in data:
    print(">>> GOOD NEWS: Camera data IS inside the chunk!")
else:
    print(">>> BAD NEWS: Camera data is missing. We must side-load it.")
