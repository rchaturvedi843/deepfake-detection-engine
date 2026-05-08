import os, shutil
from pathlib import Path

# ── YOUR EXACT PATHS ──────────────────────────────────────
ff_faces_root = r"C:\Users\sudhir chaturvedi\Desktop\deepfake dataset\ff_faces"
output_root   = r"C:\Users\sudhir chaturvedi\Desktop\deepfake dataset\deepfake codes\dataset"

# ff_faces has Fake/ and Real/ directly — no subfolders
splits = {"train": 1000, "val": 200}

for split, limit in splits.items():

    # ── FAKE images ──────────────────────────────────────
    fake_src = Path(ff_faces_root) / "Fake"          # capital F
    fake_out = Path(output_root) / split / "fake"    # lowercase in dataset
    fake_out.mkdir(parents=True, exist_ok=True)

    fake_imgs = list(fake_src.rglob("*.jpg")) + \
                list(fake_src.rglob("*.png")) + \
                list(fake_src.rglob("*.jpeg"))

    copied = 0
    for img in fake_imgs[:limit]:
        shutil.copy2(img, fake_out / img.name)
        copied += 1
    print(f"  {split}/fake: {copied} images copied")

    # ── REAL images ──────────────────────────────────────
    real_src = Path(ff_faces_root) / "Real"          # capital R
    real_out = Path(output_root) / split / "real"    # lowercase in dataset
    real_out.mkdir(parents=True, exist_ok=True)

    real_imgs = list(real_src.rglob("*.jpg")) + \
                list(real_src.rglob("*.png")) + \
                list(real_src.rglob("*.jpeg"))

    copied = 0
    for img in real_imgs[:limit]:
        shutil.copy2(img, real_out / img.name)
        copied += 1
    print(f"  {split}/real: {copied} images copied")

print("\nDataset ready!")
print(f"\nCheck your dataset folder at:\n{output_root}")