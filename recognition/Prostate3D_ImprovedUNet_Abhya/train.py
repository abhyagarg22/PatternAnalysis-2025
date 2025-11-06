# train.py — Project 7 (Abhya)
# Training script for Improved 3D UNet model on Rangpur GPU cluster

import os
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, random_split
from dataset import HipMRIDataset
from modules import ImprovedUNet3D
from torch.cuda.amp import autocast, GradScaler
import torch.nn.functional as F 



# ----------------------------
# CONFIGURATION
# ----------------------------
MRI_DIR = "/home/groups/comp3710/HipMRI_Study_open/semantic_MRs"
LABEL_DIR = "/home/groups/comp3710/HipMRI_Study_open/semantic_labels_only"

EPOCHS = 10    # Increase if GPU allows
BATCH_SIZE = 2
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

# 80% train, 20% val
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_set, val_set = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
val_loader = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=2, pin_memory=True)


# ----------------------------
# MODEL, LOSS, OPTIMIZER
# ----------------------------
model = ImprovedUNet3D().to(DEVICE)

# ← BETTER: Combined loss function
class DiceCELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.ce = nn.CrossEntropyLoss()
    
    def forward(self, pred, target):
        ce_loss = self.ce(pred, target)
        
        # Dice loss
        pred_soft = torch.softmax(pred, dim=1)
        dice_loss = 0
        for c in range(pred.shape[1]):
            pred_c = pred_soft[:, c]
            target_c = (target == c).float()
            intersection = (pred_c * target_c).sum()
            union = pred_c.sum() + target_c.sum()
            dice_loss += 1 - (2 * intersection + 1e-6) / (union + 1e-6)
        dice_loss /= pred.shape[1]
        
        return ce_loss + dice_loss

criterion = DiceCELoss()
optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='max', factor=0.5, patience=3
)
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
scaler = GradScaler(enabled=(DEVICE.type == "cuda"))
for epoch in range(EPOCHS):
    model.train()
    epoch_loss = 0.0
    epoch_dice = 0.0

    for batch_idx, (img, label) in enumerate(train_loader):
        img = img.to(DEVICE)
        label = label.to(DEVICE).long()  # important for CrossEntropyLoss

        optimizer.zero_grad()
        with autocast(enabled=(DEVICE.type == "cuda")):
            output = model(img)
            if output.shape[2:] != label.shape[1:]:
                label = F.interpolate(
                    label.unsqueeze(1).float(),
                    size=output.shape[2:],
                    mode="nearest"
                ).squeeze(1).long()
            loss = criterion(output, label)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

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
    val_dice_avg = val_dice / len(val_loader)
    print(f"Validation Dice: {val_dice / len(val_loader):.4f}\n")
    scheduler.step(val_dice_avg)
    

# ----------------------------
# SAVE MODEL
# ----------------------------
os.makedirs("models", exist_ok=True)
torch.save(model.state_dict(), "models/improved_unet3d.pth")
print("Training complete! Model saved to models/improved_unet3d.pth")
