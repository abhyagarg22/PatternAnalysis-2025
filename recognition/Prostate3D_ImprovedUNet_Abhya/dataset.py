# dataset.py — Project 7 (Abhya)
# Loads 3D MRI volumes and returns them as PyTorch tensors

import os
import torch
import nibabel as nib
from torch.utils.data import Dataset
import numpy as np

class HipMRIDataset(Dataset):
    def __init__(self, image_dir, label_dir, transform=None, crop=True):
        self.crop=crop
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
            pid = "_".join(img.split("_")[:2])  # e.g. "D031"
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

        # normalize image (keep yours)
        image = (image - np.mean(image)) / (np.std(image) + 1e-8)
        image = np.clip(image, -3, 3)
        image = (image - image.min()) / (image.max() - image.min() + 1e-8)
        if np.random.rand() > 0.5:
            image = np.flip(image, axis=1).copy()
            label = np.flip(label, axis=1).copy()
        
        if np.random.rand() > 0.5:
            image = np.flip(image, axis=2).copy()
            label = np.flip(label, axis=2).copy()

        # image: add channel
        image = np.expand_dims(image, axis=0)
        if self.crop:

            image = image[:, :, image.shape[2] // 2 - 32 : image.shape[2] // 2 + 32]
            label = label[:, :, label.shape[2] // 2 - 32 : label.shape[2] // 2 + 32]

        # label: KEEP classes 0..5
        label = label.astype(np.int64)

        image = torch.tensor(image, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.long)
        if self.transform:
            image = self.transform(image)

        return image, label

