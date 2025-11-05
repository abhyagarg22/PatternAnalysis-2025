# train.py — Project 7 (Abhya)
# Training script for Improved 3D UNet model on Rangpur GPU cluster

import os
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, random_split
from dataset import HipMRIDataset
from modules import ImprovedUNet3D
import torch.nn.functional as F 


# ----------------------------
# CONFIGURATION
# ----------------------------
MRI_DIR = "/home/groups/comp3710/HipMRI_Study_open/semantic_MRs"
LABEL_DIR = "/home/groups/comp3710/HipMRI_Study_open/semantic_labels_only"

EPOCHS = 13    # Increase if GPU allows
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
model = ImprovedUNet3D().to(DEVICE)       # outputs 6 channels now
criterion = nn.CrossEntropyLoss()         # multiclass loss
optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)

def multiclass_dice(pred, target, num_classes=6, eps=1e-6):
    """Compute mean Dice coefficient across all classes."""
    dice_scores = []
    for c in range(num_classes):
        pred_c = (pred == c).float()
        target_c = (target == c).float()
        intersection = (pred_c * target_c).sum()
        union = pred_c.sum() + target_c.sum()
        dice = (2 * intersection + eps) / (union + eps)
        dice_scores.append(dice)
    return torch.mean(torch.stack(dice_scores))


# ----------------------------
# TRAINING LOOP
# ----------------------------
for epoch in range(EPOCHS):
    model.train()
    epoch_loss = 0.0
    epoch_dice = 0.0

    for batch_idx, (img, label) in enumerate(train_loader):
        img = img.to(DEVICE)
        label = label.to(DEVICE).long()  # important for CrossEntropyLoss

        optimizer.zero_grad()
        output = model(img)  # [B, 6, D, H, W]

        # adjust label size if mismatch
        if output.shape[2:] != label.shape[1:]:
            label = F.interpolate(
                label.unsqueeze(1).float(),
                size=output.shape[2:],
                mode="nearest"
            ).squeeze(1).long()

        loss = criterion(output, label)
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            preds = torch.argmax(output, dim=1)
            dice = multiclass_dice(preds, label)

        epoch_loss += loss.item()
        epoch_dice += dice.item()

        print(f"Epoch [{epoch+1}/{EPOCHS}] Batch [{batch_idx+1}/{len(train_loader)}] "
              f"Loss: {loss.item():.4f} | Dice: {dice.item():.4f}")

    # summary for the epoch
    avg_loss = epoch_loss / len(train_loader)
    avg_dice = epoch_dice / len(train_loader)
    print(f"\n📘 Epoch [{epoch+1}/{EPOCHS}] Train Loss: {avg_loss:.4f} | Train Dice: {avg_dice:.4f}")

    # ---- VALIDATION ----
    model.eval()
    val_dice = 0.0
    with torch.no_grad():
        for img, label in val_loader:
            img = img.to(DEVICE)
            label = label.to(DEVICE).long()
            output = model(img)

            if output.shape[2:] != label.shape[1:]:
                label = F.interpolate(
                    label.unsqueeze(1).float(),
                    size=output.shape[2:],
                    mode="nearest"
                ).squeeze(1).long()

            preds = torch.argmax(output, dim=1)
            val_dice += multiclass_dice(preds, label).item()

    print(f"🧪 Validation Dice: {val_dice / len(val_loader):.4f}\n")


# ----------------------------
# SAVE MODEL
# ----------------------------
os.makedirs("models", exist_ok=True)
torch.save(model.state_dict(), "models/improved_unet3d.pth")
print("Training complete! Model saved to models/improved_unet3d.pth")
