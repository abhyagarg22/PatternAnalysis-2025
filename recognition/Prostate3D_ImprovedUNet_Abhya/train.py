# train.py — Project 7 (Abhya)
# Training script for Improved 3D UNet model

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from dataset import HipMRIDataset
from modules import ImprovedUNet3D
import os

# ----------------------------
# CONFIGURATION
# ----------------------------
DATA_DIR = '.'  # change to real dataset path on Rangpur
EPOCHS = 2      # keep small for local testing
BATCH_SIZE = 1
LR = 0.001
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ----------------------------
# DATASET & DATALOADER
# ----------------------------
dataset = HipMRIDataset(DATA_DIR)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# ----------------------------
# MODEL, LOSS, OPTIMIZER
# ----------------------------
model = ImprovedUNet3D().to(DEVICE)
criterion = nn.BCELoss()  # simple loss for segmentation
optimizer = optim.Adam(model.parameters(), lr=LR)

# ----------------------------
# TRAINING LOOP
# ----------------------------
for epoch in range(EPOCHS):
    model.train()
    epoch_loss = 0.0

    for batch_idx, img in enumerate(dataloader):
        img = img.to(DEVICE)
        optimizer.zero_grad()
        output = model(img)
        loss = criterion(output, img)  # using input as target (fake data)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        print(f"Epoch [{epoch+1}/{EPOCHS}] Batch [{batch_idx+1}] Loss: {loss.item():.4f}")

    print(f"Epoch [{epoch+1}/{EPOCHS}] Average Loss: {epoch_loss/len(dataloader):.4f}")

# ----------------------------
# SAVE MODEL
# ----------------------------
os.makedirs('models', exist_ok=True)
torch.save(model.state_dict(), 'models/improved_unet3d.pth')
print("Training complete! Model saved to models/improved_unet3d.pth")

