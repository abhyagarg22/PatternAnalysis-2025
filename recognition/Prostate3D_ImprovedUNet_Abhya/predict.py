# predict.py — Project 7 (Abhya)
# Inference script for trained Improved 3D U-Net model

import torch
import nibabel as nib
import numpy as np
from modules import ImprovedUNet3D
from dataset import HipMRIDataset
import os

MODEL_PATH = 'models/improved_unet3d.pth'
DATA_DIR = "/home/groups/comp3710/HipMRI_Study_open/semantic_MRs"  # change to test data path
LABEL_DIR = "/home/groups/comp3710/HipMRI_Study_open/semantic_labels_only"
SAVE_DIR = 'predictions'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

os.makedirs(SAVE_DIR, exist_ok=True)

# Load model
model = ImprovedUNet3D().to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

# Load data 
dataset = HipMRIDataset(DATA_DIR, LABEL_DIR)

if len(dataset) == 0:
    raise RuntimeError(f"No MRI files found in {DATA_DIR}")
# if len(dataset) == 0:
    #print("No MRI files found — creating fake test volume.")
    #fake = np.random.rand(64, 64, 64)
    #nib.save(nib.Nifti1Image(fake, np.eye(4)), 'fake_test.nii')
    #dataset = HipMRIDataset('.')

img, _ = dataset[0]
img = img.unsqueeze(0).to(DEVICE)  # [1, 1, D, H, W]

with torch.no_grad():
    pred = model(img)
    pred = (pred > 0.5).float()

pred_np = pred.squeeze().cpu().numpy()
save_path = os.path.join(SAVE_DIR, 'prediction.nii')
nib.save(nib.Nifti1Image(pred_np, np.eye(4)), save_path)
print(f"Prediction saved to {save_path}")
print("Output shape:", pred_np.shape)
