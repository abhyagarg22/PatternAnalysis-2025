# dataset.py — Project 7 (Abhya)
# Loads 3D MRI volumes and returns them as PyTorch tensors

import os
import torch
import nibabel as nib
from torch.utils.data import Dataset
import numpy as np

class HipMRIDataset(Dataset):
    def __init__(self, image_dir, label_dir, transform=None):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.transform = transform

        # Match images and labels based on patient ID prefix
        self.image_files = sorted([
            f for f in os.listdir(image_dir) if f.endswith('.nii.gz')
        ])
        self.label_files = sorted([
            f for f in os.listdir(label_dir) if f.endswith('.nii.gz')
        ])

        # Filter to keep only those with matching IDs
        self.pairs = []
        for img in self.image_files:
            pid = img.split('_')[0]  # e.g. "D031"
            match = next((l for l in self.label_files if l.startswith(pid)), None)
            if match:
                self.pairs.append((img, match))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img_name, label_name = self.pairs[idx]
        img_path = os.path.join(self.image_dir, img_name)
        label_path = os.path.join(self.label_dir, label_name)

        # Load MRI and label
        image = nib.load(img_path).get_fdata()
        label = nib.load(label_path).get_fdata()

        # Normalize and convert
        # Normalize both image and label
        image = (image - np.min(image)) / (np.max(image) - np.min(image) + 1e-8)
        label = (label - np.min(label)) / (np.max(label) - np.min(label) + 1e-8)

        # Add channel dimension
        image = np.expand_dims(image, axis=0)
        label = np.expand_dims(label, axis=0)


        image = torch.tensor(image, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.float32)

        if self.transform:
            image = self.transform(image)

        return image, label
