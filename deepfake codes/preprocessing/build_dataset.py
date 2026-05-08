import os
import shutil
import random

# =========================
# DATASET PATHS
# =========================

archive2 = r"C:\Users\sudhir chaturvedi\Desktop\deepfake dataset\archive (2)\Dataset"
faces = r"C:\Users\sudhir chaturvedi\Desktop\deepfake dataset\ff_faces"

final = r"C:\Users\sudhir chaturvedi\Desktop\deepfake dataset\final_dataset"

train_ratio = 0.7
val_ratio = 0.15
test_ratio = 0.15


# =========================
# CREATE FOLDER STRUCTURE
# =========================

folders = [
    os.path.join(final,"Train","Real"),
    os.path.join(final,"Train","Fake"),
    os.path.join(final,"Validation","Real"),
    os.path.join(final,"Validation","Fake"),
    os.path.join(final,"Test","Real"),
    os.path.join(final,"Test","Fake"),
]

for folder in folders:
    os.makedirs(folder, exist_ok=True)


# =========================
# COLLECT IMAGES
# =========================

def collect_images(folder):

    images = []

    for root, dirs, files in os.walk(folder):
        for file in files:
            if file.lower().endswith((".jpg",".jpeg",".png")):
                images.append(os.path.join(root,file))

    return images


# =========================
# SPLIT DATASET
# =========================

def split_and_copy(images,label):

    random.shuffle(images)

    total = len(images)

    train_end = int(total * train_ratio)
    val_end = int(total * (train_ratio + val_ratio))

    train = images[:train_end]
    val = images[train_end:val_end]
    test = images[val_end:]


    for img in train:
        shutil.copy(img, os.path.join(final,"Train",label))

    for img in val:
        shutil.copy(img, os.path.join(final,"Validation",label))

    for img in test:
        shutil.copy(img, os.path.join(final,"Test",label))


print("Collecting Real images...")

real_images = collect_images(os.path.join(archive2,"Train","Real")) \
            + collect_images(os.path.join(faces,"Real"))


print("Collecting Fake images...")

fake_images = collect_images(os.path.join(archive2,"Train","Fake")) \
            + collect_images(os.path.join(faces,"Fake"))


print("Splitting dataset...")

split_and_copy(real_images,"Real")
split_and_copy(fake_images,"Fake")


print("\nDataset build complete")
