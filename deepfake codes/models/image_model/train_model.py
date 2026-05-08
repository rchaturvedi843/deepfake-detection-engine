"""
train_model.py  —  Debugged & Advanced
=======================================
BUGS FIXED:
  1. ✅ No validation accuracy printed during training
  2. ✅ No early stopping (wastes time training past best epoch)
  3. ✅ CrossEntropyLoss on imbalanced data (use weighted loss)
  4. ✅ Model saved as raw state_dict (no metadata saved)
  5. ✅ Learning rate never changes (add scheduler)
  6. ✅ Class mapping not verified (can cause inverted results)

NEW FEATURES:
  + Weighted CrossEntropyLoss for class imbalance
  + EarlyStopping (patience=5)
  + ReduceLROnPlateau scheduler
  + Per-epoch validation accuracy + F1 score
  + Saves best model with metadata (epoch, val_acc, class_names)
  + Prints class mapping so you always know Fake=0, Real=1
  + Supports CLI args for Colab training
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import f1_score, confusion_matrix
import os
import random
import time
import argparse
import json
from pathlib import Path

# ── CLI args (for Colab) ─────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--data_root", type=str,
    default=r"C:\Users\sudhir chaturvedi\Desktop\deepfake dataset\final_dataset")
parser.add_argument("--arch",       type=str, default="resnet18",
    choices=["resnet18", "resnet34", "resnet50"])
parser.add_argument("--epochs",     type=int, default=25)
parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--lr",         type=float, default=0.0001)
parser.add_argument("--output_dir", type=str, default="checkpoints")
parser.add_argument("--num_workers",type=int, default=0)
parser.add_argument("--max_train",  type=int, default=40000)
parser.add_argument("--max_val",    type=int, default=8000)
args = parser.parse_args()

Path(args.output_dir).mkdir(parents=True, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\n🖥️  Device: {device}")
print(f"📂  Data:   {args.data_root}")
print(f"🧠  Arch:   {args.arch}")
print(f"⚙️   Epochs: {args.epochs}  |  Batch: {args.batch_size}  |  LR: {args.lr}\n")

# ── Transforms ───────────────────────────────────────────
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(p=0.1),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
    transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    transforms.RandomErasing(p=0.15),
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# ── Datasets ─────────────────────────────────────────────
def limit(ds, n):
    idx = list(range(len(ds)))
    random.shuffle(idx)
    return Subset(ds, idx[:n])

train_ds = datasets.ImageFolder(os.path.join(args.data_root, "Train"),  train_transform)
val_ds   = datasets.ImageFolder(os.path.join(args.data_root, "Validation"), val_transform)

# ── IMPORTANT: print class mapping ───────────────────────
print("📋 Class mapping (ImageFolder sorts alphabetically):")
for cls, idx in train_ds.class_to_idx.items():
    print(f"   class {idx} = {cls}")
print()
# Expected: class 0 = Fake, class 1 = Real
# This confirms probs[0]=Fake, probs[1]=Real in predictor

train_ds = limit(train_ds, args.max_train)
val_ds   = limit(val_ds,   args.max_val)

print(f"Train: {len(train_ds)}  |  Val: {len(val_ds)}")

# ── Class weights for imbalanced data ────────────────────
labels = [train_ds.dataset.targets[i] for i in train_ds.indices]
n_fake = labels.count(0)
n_real = labels.count(1)
w_fake = len(labels) / (2 * n_fake) if n_fake > 0 else 1.0
w_real = len(labels) / (2 * n_real) if n_real > 0 else 1.0
class_weights = torch.tensor([w_fake, w_real]).to(device)
print(f"⚖️  Class weights: Fake={w_fake:.3f}  Real={w_real:.3f}\n")

# ── Loaders ──────────────────────────────────────────────
train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                          shuffle=True, num_workers=args.num_workers)
val_loader   = DataLoader(val_ds,   batch_size=args.batch_size,
                          shuffle=False, num_workers=args.num_workers)

# ── Model ────────────────────────────────────────────────
arch_map = {
    "resnet18": (models.resnet18, models.ResNet18_Weights.DEFAULT),
    "resnet34": (models.resnet34, models.ResNet34_Weights.DEFAULT),
    "resnet50": (models.resnet50, models.ResNet50_Weights.DEFAULT),
}
model_fn, weights = arch_map[args.arch]
model = model_fn(weights=weights)

# Freeze early layers
for param in model.parameters():
    param.requires_grad = False

# Unfreeze last 2 blocks + FC
for param in model.layer3.parameters():
    param.requires_grad = True
for param in model.layer4.parameters():
    param.requires_grad = True

model.fc = nn.Sequential(
    nn.Dropout(0.5),
    nn.Linear(model.fc.in_features, 2)
)
model = model.to(device)

# ── Loss, Optimizer, Scheduler ───────────────────────────
criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
optimizer = optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=args.lr, weight_decay=1e-4
)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="max", factor=0.5, patience=3, verbose=True
)

# ── Early Stopping ───────────────────────────────────────
class EarlyStopping:
    def __init__(self, patience=5):
        self.patience = patience
        self.best = 0.0
        self.counter = 0
        self.stop = False

    def step(self, val_acc):
        if val_acc > self.best + 0.001:
            self.best = val_acc
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.stop = True

early_stop = EarlyStopping(patience=5)
best_val_acc = 0.0
history = []
start = time.time()

print("=" * 60)
print("  TRAINING STARTED")
print("=" * 60)

for epoch in range(1, args.epochs + 1):

    # ── Train ────────────────────────────────────────────
    model.train()
    train_loss = train_correct = train_total = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        train_loss += loss.item()
        preds = outputs.argmax(1)
        train_correct += (preds == labels).sum().item()
        train_total   += labels.size(0)

    train_acc = 100 * train_correct / train_total

    # ── Validate ─────────────────────────────────────────
    model.eval()
    val_loss = val_correct = val_total = 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss    += loss.item()
            preds = outputs.argmax(1)
            val_correct += (preds == labels).sum().item()
            val_total   += labels.size(0)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    val_acc = 100 * val_correct / val_total
    f1 = f1_score(all_labels, all_preds, average="weighted", zero_division=0) * 100

    print(f"Epoch {epoch:02d}/{args.epochs} | "
          f"Train Loss={train_loss/len(train_loader):.4f} Acc={train_acc:.1f}% | "
          f"Val Loss={val_loss/len(val_loader):.4f} Acc={val_acc:.1f}% F1={f1:.1f}% | "
          f"LR={optimizer.param_groups[0]['lr']:.6f}")

    history.append({
        "epoch": epoch,
        "train_acc": round(train_acc, 2),
        "val_acc":   round(val_acc, 2),
        "f1":        round(f1, 2),
    })

    scheduler.step(val_acc)

    # Save best checkpoint with metadata
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        ckpt_path = os.path.join(args.output_dir, "best_model.pth")
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_acc": round(val_acc, 2),
            "f1": round(f1, 2),
            "arch": args.arch,
            "class_to_idx": train_ds.dataset.class_to_idx,
            "config": vars(args),
        }, ckpt_path)
        print(f"  ✅ Saved best model → val_acc={val_acc:.1f}%")

    early_stop.step(val_acc)
    if early_stop.stop:
        print(f"\n⏹️  Early stopping at epoch {epoch} (no improvement for 5 epochs)")
        break

# ── Save history ─────────────────────────────────────────
with open(os.path.join(args.output_dir, "history.json"), "w") as f:
    json.dump(history, f, indent=2)

elapsed = (time.time() - start) / 60
print(f"\n{'='*60}")
print(f"  TRAINING COMPLETE")
print(f"  Best Val Accuracy : {best_val_acc:.2f}%")
print(f"  Total Time        : {elapsed:.1f} minutes")
print(f"  Model saved to    : {args.output_dir}/best_model.pth")
print(f"{'='*60}")