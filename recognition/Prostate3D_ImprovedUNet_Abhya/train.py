# train.py — Project 7 (Abhya)
# Training script for Improved 3D UNet model on Rangpur GPU cluster

import os
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from dataset import HipMRIDataset
from modules import ImprovedUNet3D
import torch.nn.functional as F 
def dice_loss(pred, target, smooth=1e-5):
    pred = torch.sigmoid(pred)
    intersection = (pred * target).sum()
    return 1 - (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)

def combined_loss(pred, target):
    bce = F.binary_cross_entropy_with_logits(pred, target)
    dice = dice_loss(pred, target)
    return bce + dice

# ----------------------------
# CONFIGURATION
# ----------------------------
MRI_DIR = "/home/groups/comp3710/HipMRI_Study_open/semantic_MRs"
LABEL_DIR = "/home/groups/comp3710/HipMRI_Study_open/semantic_labels_only"

EPOCHS = 5      # Increase if GPU allows
BATCH_SIZE = 1
LR = 0.001
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

dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# ----------------------------
# MODEL, LOSS, OPTIMIZER
# ----------------------------
model = ImprovedUNet3D().to(DEVICE)
criterion = combined_loss 
optimizer = optim.Adam(model.parameters(), lr=LR)

# ----------------------------
# TRAINING LOOP
# ----------------------------
for epoch in range(EPOCHS):
    model.train()
    epoch_loss = 0.0

    for batch_idx, batch in enumerate(dataloader):
        if isinstance(batch, (list, tuple)):
            img, label = batch
        else:
            img, label = batch, batch

        img = img.to(DEVICE)
        label = label.to(DEVICE)
        optimizer.zero_grad()
        output = model(img)
        if output.shape != label.shape:
          label = torch.nn.functional.interpolate(
            label, size=output.shape[2:], mode='trilinear', align_corners=False
          )
        loss = criterion(output, label)
        # Compute Dice metric for monitoring
        with torch.no_grad():
          preds = torch.sigmoid(output)
          preds = (preds > 0.5).float()
          intersection = (preds * label).sum()
          dice = (2. * intersection) / (preds.sum() + label.sum() + 1e-8)
        print(f"Epoch [{epoch+1}/{EPOCHS}] Batch [{batch_idx+1}/{len(dataloader)}] Loss: {loss.item():.4f} | Dice: {dice.item():.4f}")

        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        print(f"Epoch [{epoch+1}/{EPOCHS}] Batch [{batch_idx+1}/{len(dataloader)}] Loss: {loss.item():.4f}")

    avg_loss = epoch_loss / len(dataloader)
    print(f"Epoch [{epoch+1}/{EPOCHS}] Average Loss: {avg_loss:.4f}")

# ----------------------------
# SAVE MODEL
# ----------------------------
os.makedirs("models", exist_ok=True)
torch.save(model.state_dict(), "models/improved_unet3d.pth")
print("Training complete! Model saved to models/improved_unet3d.pth")
