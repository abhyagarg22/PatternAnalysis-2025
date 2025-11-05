# train.py — Project 7 (Abhya)
# Training script for Improved 3D UNet model on Rangpur GPU cluster

import os
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, random_split
from dataset import HipMRIDataset
from modules import ImprovedUNet3D
import torch.nn.functional as F 
def dice_loss(pred, target, smooth=1e-5):
    pred = torch.sigmoid(pred)
    intersection = (pred * target).sum()
    return 1 - (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)

def focal_loss(pred, target, alpha=0.8, gamma=2.0):
    bce = F.binary_cross_entropy_with_logits(pred, target, reduction="none")
    pt = torch.exp(-bce)
    return (alpha * (1 - pt) ** gamma * bce).mean()

def combined_loss(pred, target):
    # use Dice + Focal to push outputs closer to 0 or 1
    return dice_loss(pred, target) + focal_loss(pred, target)


# ----------------------------
# CONFIGURATION
# ----------------------------
MRI_DIR = "/home/groups/comp3710/HipMRI_Study_open/semantic_MRs"
LABEL_DIR = "/home/groups/comp3710/HipMRI_Study_open/semantic_labels_only"

EPOCHS = 10    # Increase if GPU allows
BATCH_SIZE = 1
LR = 0.0001
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f"Using device: {DEVICE}")
if DEVICE.type == "cuda":
    print("CUDA available — training on GPU")
else:
    print("CUDA not available — training on CPU")

# ----------------------------
# DATASET & DATALOADER
# ----------------------------
dataset = HipMRIDataset(MRI_DIR, LABEL_DIR, transform=None)
if len(dataset) == 0:
    raise RuntimeError(f"No .nii.gz files found in {MRI_DIR}. Please check dataset path.")

# 80% train, 20% val
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_set, val_set = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_set, batch_size=1, shuffle=False)


# ----------------------------
# MODEL, LOSS, OPTIMIZER
# ----------------------------
model = ImprovedUNet3D().to(DEVICE)
criterion = combined_loss 
optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)

# ----------------------------
# TRAINING LOOP
# ----------------------------
for epoch in range(EPOCHS):
    model.train()
    epoch_loss = 0.0
    epoch_dice = 0.0

    for batch_idx, (img, label) in enumerate(train_loader):
        img = img.to(DEVICE)
        label = label.to(DEVICE)

        optimizer.zero_grad()
        output = model(img)
        if output.shape != label.shape:
            label = F.interpolate(label, size=output.shape[2:], mode='trilinear', align_corners=False)

        loss = criterion(output, label)

        # Dice coefficient
        with torch.no_grad():
            preds = torch.sigmoid(output)
            preds = (preds > 0.5).float()
            intersection = (preds * label).sum()
            dice = (2. * intersection) / (preds.sum() + label.sum() + 1e-8)
            print(f"    batch pred mean={preds.mean().item():.4f}")

        epoch_loss += loss.item()
        epoch_dice += dice.item()

        print(f"Epoch [{epoch+1}/{EPOCHS}] Batch [{batch_idx+1}/{len(train_loader)}] Loss: {loss.item():.4f} | Dice: {dice.item():.4f}")

        loss.backward()
        optimizer.step()

    avg_loss = epoch_loss / len(train_loader)
    avg_dice = epoch_dice / len(train_loader)
    print(f"Epoch [{epoch+1}/{EPOCHS}] Train Loss: {avg_loss:.4f} | Train Dice: {avg_dice:.4f}")
    # --- VALIDATION LOOP ---
    model.eval()
    val_dice = 0.0
    with torch.no_grad():
        for img, label in val_loader:
            img = img.to(DEVICE)
            label = label.to(DEVICE)

            output = model(img)
        # make sure label matches output size
            if output.shape != label.shape:
                label = F.interpolate(label, size=output.shape[2:], mode='trilinear', align_corners=False)

            output = torch.sigmoid(output)
            pred_bin = (output > 0.5).float()
            intersection = (pred_bin * label).sum()
            dice = (2. * intersection) / (pred_bin.sum() + label.sum() + 1e-8)
            val_dice += dice.item()

    val_dice = val_dice / len(val_loader)
    print(f"Validation Dice: {val_dice:.4f}")
    model.train()


# ----------------------------
# SAVE MODEL
# ----------------------------
os.makedirs("models", exist_ok=True)
torch.save(model.state_dict(), "models/improved_unet3d.pth")
print("Training complete! Model saved to models/improved_unet3d.pth")
