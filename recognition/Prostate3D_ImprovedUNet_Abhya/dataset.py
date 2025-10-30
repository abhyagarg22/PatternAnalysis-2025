# dataset.py — Project 7 (Abhya)
# Loads 3D MRI volumes and returns them as PyTorch tensors

import os
import torch
import nibabel as nib
from torch.utils.data import Dataset
import numpy as np

class HipMRIDataset(Dataset):
    """Dataset class for 3D MRI volumes (.nii / .nii.gz)"""
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.image_paths = sorted([
            os.path.join(data_dir, f)
            for f in os.listdir(data_dir)
            if f.endswith('.nii') or f.endswith('.nii.gz')
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = nib.load(img_path).get_fdata()

        # Normalize to [0, 1]
        image = (image - np.min(image)) / (np.max(image) - np.min(image) + 1e-8)

        # Add channel dimension [1, D, H, W]
        image = np.expand_dims(image, axis=0)
        image = torch.tensor(image, dtype=torch.float32)

        if self.transform:
            image = self.transform(image)

        return image
