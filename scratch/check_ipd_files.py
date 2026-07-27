import torch
import glob
from pathlib import Path

corrupted = []
files = glob.glob("data/datasets/*/features/IPD*/**/*.pt", recursive=True)
for f in files:
    try:
        torch.load(f, weights_only=False)
    except Exception as e:
        print(f"Corrupted: {f} ({e})")
        corrupted.append(f)

print(f"Found {len(corrupted)} corrupted files.")
with open("scratch/corrupted_files.txt", "w") as f_out:
    for c in corrupted:
        f_out.write(c + "\n")
