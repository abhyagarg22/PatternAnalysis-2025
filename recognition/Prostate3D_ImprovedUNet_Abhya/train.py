# train.py — Project 7 (Abhya)
# Training script for Improved 3D U-Net

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from dataset import HipMRIDataset
from modules import ImprovedUNet3D
import os

# ----------------------------
# CONFIG
# ----------------------------
DATA_DIR = '/home/groups/comp3710/HipMRI_Study_open'  # change to dataset path on Rangpur
EPOCHS = 5
BATCH_SIZE = 1
LR = 1e-3
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ----------------------------
# LOAD DATA
# ----------------------------
dataset = HipMRIDataset(DATA_DIR)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# ----------------------------
# MODEL, LOSS, OPTIMIZER
# ----------------------------
model = ImprovedUNet3D().to(DEVICE)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

# ----------------------------
# TRAINING LOOP
# ----------------------------
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0.0

    for batch_idx, img in enumerate(dataloader):
        img = img.to(DEVICE)
        optimizer.zero_grad()
        output = model(img)
        loss = criterion(output, img)  # using input as pseudo-target for testing
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        print(f"Epoch [{epoch+1}/{EPOCHS}] Batch [{batch_idx+1}] Loss: {loss.item():.4f}")

    print(f"Epoch [{epoch+1}/{EPOCHS}] Average Loss: {total_loss/len(dataloader):.4f}")

# ----------------------------
# SAVE MODEL
# ----------------------------
os.makedirs('models', exist_ok=True)
torch.save(model.state_dict(), 'models/improved_unet3d.pth')
print("Training complete! Model saved to models/improved_unet3d.pth")
