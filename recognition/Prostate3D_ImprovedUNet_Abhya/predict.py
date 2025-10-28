# predict.py — Project 7 (Abhya)
# Runs inference with the trained Improved 3D UNet model

import torch
import nibabel as nib
import numpy as np
from modules import ImprovedUNet3D
from dataset import HipMRIDataset
import os

# ----------------------------
# CONFIG
# ----------------------------
MODEL_PATH = 'models/improved_unet3d.pth'
DATA_DIR = '.'   # change to real test folder on Rangpur
SAVE_DIR = 'predictions'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

os.makedirs(SAVE_DIR, exist_ok=True)

# ----------------------------
# LOAD MODEL
# ----------------------------
model = ImprovedUNet3D().to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

# ----------------------------
# LOAD SAMPLE DATA
# ----------------------------
dataset = HipMRIDataset(DATA_DIR)
if len(dataset) == 0:
    # make a fake volume if no MRI files found
    print("No MRI files found, generating fake input for testing...")
    fake = np.random.rand(64, 64, 64)
    nib.save(nib.Nifti1Image(fake, np.eye(4)), 'fake_test.nii')
    dataset = HipMRIDataset('.')

img = dataset[0].unsqueeze(0).to(DEVICE)  # [1, 1, D, H, W]

# ----------------------------
# PREDICTION
# ----------------------------
with torch.no_grad():
    pred = model(img)
    pred = (pred > 0.5).float()  # threshold

# ----------------------------
# SAVE OUTPUT
# ----------------------------
pred_np = pred.squeeze().cpu().numpy()
nib.save(nib.Nifti1Image(pred_np, np.eye(4)), os.path.join(SAVE_DIR, 'prediction.nii'))
print("Prediction complete! Saved to", os.path.join(SAVE_DIR, 'prediction.nii'))
print("Output shape:", pred_np.shape)
