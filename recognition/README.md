# Project 7 - 3D Prostate MRI Segmentation Using Improved UNet3D  
**Author:** Abhya Garg (48299785)  
**Course:** COMP3710 - Pattern Analysis  
**Platform:** UQ Rangpur GPU Cluster (A100)  
**Date:** 7th November 2025  

---

## Objective
This project focuses on **multi-class 3D prostate MRI segmentation** using an **Improved 3D U-Net** model.  
The goal is to automatically segment anatomical structures (e.g., prostate zones, bladder, rectum) from MRI volumes.  

The model extends the improved 3D U-Net with:
- **Residual connections** for stable training  
- **Dropout regularization** for better generalization  
- **Multi-class output** (6 labels) trained with **Dice + Cross Entropy loss**
  
Segmentation accuracy is evaluated using the **Dice Similarity Coefficient (DSC)** per class.

---

## Model Overview

### Architecture Highlights
- **Encoder–Decoder 3D U-Net** with skip connections  
- **Residual Blocks** in both encoder and decoder paths  
- **3D Convolutions (3×3×3)** and **Instance Normalization**  
- **Dropout (0.3)** for regularization  
- **Softmax output** for 6-class segmentation  
- **Loss function:** Weighted combination of Cross Entropy + Dice Loss  

### Model Components
| Component | Description |
|------------|--------------|
| `ResidualBlock3D` | Two conv layers + BatchNorm + ReLU + skip path |
| `ImprovedUNet3D` | Encoder–decoder with skip connections, residuals, dropout |
| `DiceLoss3D` | Differentiable Dice loss supporting multiple labels |

---

## Dataset
**Source:** `/home/groups/comp3710/HipMRI_Study_open/`  
This dataset contains anonymized 3D MRI volumes with corresponding semantic labels for prostate anatomy.

| Folder | Description |
|---------|--------------|
| `semantic_MRs/` | Raw MRI volumes |
| `semantic_labels_only/` | Ground-truth label maps (0–5) |

Each `.nii.gz` file corresponds to one MRI scan and its segmentation labels (e.g., `W029_Week7_LFOV.nii.gz` and `W029_Week7_SEMANTIC.nii.gz`).

---

## Preprocessing

| Step | Description |
|------|--------------|
| Normalization | Intensity scaled to [0,1] per volume |
| Shape alignment | Automatic padding to match input shape (256×256×128) |
| Orientation check | Ensures `(L, P, S)` orientation consistency using `nibabel.orientations` |
| Tensor conversion | Converted to PyTorch `(C×D×H×W)` tensors |

---


## Problem Description

Manual prostate segmentation from MRI scans is a time-consuming and error-prone task in clinical workflows.  
This project automates that process using a deep learning approach - an **Improved 3D U-Net** - to accurately identify and delineate the prostate gland in volumetric MRI data.  
By leveraging residual connections and volumetric convolutions, the network can capture subtle boundaries and anatomical features in 3D.

---

## Algorithm Explanation

The **Improved 3D U-Net** is designed to segment medical volumes like prostate MRI scans.  
It follows an **encoder–decoder** structure where:

- The **encoder** compresses the 3D MRI volume and captures spatial context using convolution + pooling layers.  
- The **decoder** reconstructs the segmentation map using transposed convolutions and skip connections to restore details.  
- **Residual blocks** allow the model to learn deeper features without vanishing gradients.  
- **Dropout layers** improve generalization by preventing overfitting.  
- The **Dice + BCE combined loss** optimizes both region overlap and pixel-wise accuracy, ensuring smoother prostate boundaries.

This architecture efficiently identifies prostate regions within noisy MRI data, achieving accurate segmentation while maintaining training stability.


### Key Classes
- `ResidualBlock3D`: two 3D convolutions + BatchNorm + ReLU + Dropout + skip connection  
- `ImprovedUNet3D`: full encoder-decoder with ConvTranspose3D upsampling  

---

## Dataset
**Source:** `/home/groups/comp3710/HipMRI_Study_open/`

| Type | Folder |
|------|---------|
| MRI Volumes | `semantic_MRs/` |
| Ground-truth Labels | `semantic_labels_only/` |

Each `.nii.gz` file represents a 3D MRI volume.  
Pairs are matched by patient ID (e.g., `B040_Week0_SEMANTIC.nii.gz`).  
All volumes are normalized to [0, 1] and loaded as tensors `(1 × D × H × W)`.

---

## Preprocessing & Data Splits

### Preprocessing Steps
1. **Normalization:**  
   Each MRI and label volume is intensity-normalized to the range [0, 1] to ensure consistent contrast across scans.  
   Formula:  
   \[
   I_{norm} = \frac{I - \min(I)}{\max(I) - \min(I) + 1e-8}
   \]
2. **Channel Expansion:**  
   Each 3D volume is expanded to include a channel dimension `(1 × D × H × W)` for PyTorch compatibility.  
3. **Voxel Alignment:**  
   Both MRI and label volumes are spatially matched by patient ID prefix (e.g., `B040_Week0_SEMANTIC.nii.gz`).  

### Data Split Justification
- The dataset contains multiple 3D MRI volumes from distinct patients.  
- Since the dataset size is limited and the focus is on model architecture behavior, a **single unified dataset** was used for 15-epoch training to maximize data exposure.  
- A **validation Dice coefficient** was computed per batch to monitor convergence stability.  
- For larger-scale experiments, an 80-10-10 train/validation/test split would be recommended, ensuring patient-wise separation to prevent data leakage.

---

## Implementation Files
| File | Purpose |
|------|----------|
| `dataset.py` | Loads and normalizes MRI and label volumes |
| `modules.py` | Defines the Improved 3D U-Net architecture |
| `train.py` | Trains the model and prints loss & Dice metrics |
| `predict.py` | Runs inference and saves predictions |
| `train_gpu.slurm` | SLURM batch script for GPU training |

---

## Dependencies & Environment

To ensure reproducibility, the following setup was used on the **UQ Rangpur GPU Cluster**:

| Package | Version |
|----------|----------|
| Python | 3.10 |
| PyTorch | 2.1.0 |
| Torchvision | 0.16.0 |
| Numpy | 1.26 |
| Matplotlib | 3.8 |
| Nibabel | 5.2 |
| CUDA | 12.1 |

---

## Training Procedure
### Submit Job
```bash
sbatch train_gpu.slurm
```
---
## Training Configuration

| Parameter       | Value          |
|-----------------|----------------|
| Epochs          | 15              |
| Batch Size      | 1              |
| Learning Rate   | 0.0005          |
| Optimizer       | Adam           |
| Loss Function   | BCE + Dice     |
| Device          | CUDA (A100 GPU) |

---

## Output (Log)

<img width="829" height="379" alt="image" src="https://github.com/user-attachments/assets/364ad08a-41dc-4c4e-916e-cfc37ba630e4" />


---

## Trained Model

After training, the model weights are automatically saved to:
```bash
models/improved_unet3d.pth
```
This file contains all learned parameters from the Improved 3D U-Net model after 15 epochs of training on the Rangpur A100 GPU cluster.

---

## Inference / Prediction

Run inference after training using:
```bash
srun -p a100 --gres=gpu:a100:1 --cpus-per-task=2 --time=00:15:00 python predict.py
```
---

## Outputs

| File | Description |
|------|--------------|
| `predictions/prediction.nii.gz` | 3D predicted segmentation volume |
| `outputs` | Mid-slice visualization (optional) |

---

## Visualization

<img width="592" height="422" alt="image" src="https://github.com/user-attachments/assets/c4c24033-2cbc-4d09-a9a7-8ce016a3e1cc" />
<img width="594" height="424" alt="image" src="https://github.com/user-attachments/assets/af0ee719-cf1e-4c1a-8e9c-6c7a8522d7f9" />
<img width="897" height="265" alt="image" src="https://github.com/user-attachments/assets/63905579-be8a-434a-865c-224bbaa48204" />
<img width="195" height="197" alt="image" src="https://github.com/user-attachments/assets/f56e8372-31cd-42d3-b635-d032a421e48b" />
<img width="194" height="196" alt="image" src="https://github.com/user-attachments/assets/eb7e479e-12ad-4921-a1c7-b3904637fa26" />

### Visualization Code snippet
```bash
import os
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from scipy.ndimage import zoom

# --- Folder paths ---
pred_dir = "predictions"
label_dir = "/home/groups/comp3710/HipMRI_Study_open/semantic_labels_only"
image_dir = "/home/groups/comp3710/HipMRI_Study_open/semantic_MRs"
save_dir = "outputs/visuals_last3_fixed_dims_v2"
os.makedirs(save_dir, exist_ok=True)

def normalize_slice(slice_data):
    slice_data = np.nan_to_num(slice_data)
    slice_data -= slice_data.min()
    if slice_data.max() > 0:
        slice_data /= slice_data.max()
    return slice_data

def match_axes_and_resize(pred, mri_shape):
    """Make sure pred axes match MRI shape order, then resize."""
    if pred.shape[::-1] == mri_shape:  # sometimes reversed
        pred = np.transpose(pred, (2, 1, 0))
        print("Transposed prediction (reversed axes)")
    elif pred.shape[1:] == mri_shape[:-1]:
        pred = np.transpose(pred, (1, 0, 2))
        print("Transposed prediction (swapped XY)")
    if pred.shape != mri_shape:
        scale = np.array(mri_shape) / np.array(pred.shape)
        pred = zoom(pred, zoom=scale, order=0)
        print(f"Resized prediction from {pred.shape} → {mri_shape}")
    return pred

def make_comparison(pred_file):
    base = pred_file.replace("_prediction.nii.gz", "")
    gt_file = f"{base}_SEMANTIC.nii.gz"
    image_file = f"{base}_LFOV.nii.gz"

    pred_path = os.path.join(pred_dir, pred_file)
    gt_path = os.path.join(label_dir, gt_file)
    img_path = os.path.join(image_dir, image_file)

    if not (os.path.exists(pred_path) and os.path.exists(gt_path) and os.path.exists(img_path)):
        print(f"Skipping {pred_file} (missing files)")
        return

    # --- Load ---
    pred = nib.load(pred_path).get_fdata()
    gt = nib.load(gt_path).get_fdata()
    img = nib.load(img_path).get_fdata()

    # --- Align prediction ---
    pred = match_axes_and_resize(pred, img.shape)

    # --- Slice ---
    mid = img.shape[2] // 2
    img_slice = normalize_slice(img[:, :, mid])
    gt_slice = gt[:, :, mid]
    pred_slice = pred[:, :, mid]

    # --- Plot ---
    plt.figure(figsize=(12, 4))
    plt.suptitle(f"{base} — Segmentation Comparison", fontsize=13, fontweight="bold")

    plt.subplot(1, 3, 1)
    plt.imshow(img_slice, cmap="gray")
    plt.title("MRI Slice")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(gt_slice, cmap="viridis")
    plt.title("Ground Truth Mask")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(pred_slice, cmap="viridis")
    plt.title("Predicted Mask (Fixed)")
    plt.axis("off")

    out_path = os.path.join(save_dir, f"comparison_{base}.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=250)
    plt.close()
    print(f"Saved: {out_path}")

# --- Run on last 3 predictions ---
all_preds = sorted([f for f in os.listdir(pred_dir) if f.endswith(".nii.gz")])
for f in all_preds[-3:]:
    make_comparison(f)

```
---

## Quantitative Results

### Training Summary (Final Epoch)
| Metric | Value |
|---------|--------|
| **Average Training Loss** | **0.4554** |
| **Final Dice Coefficient** | **0.7424** |
| **Epochs Trained** | 15 |
| **Model Saved** | `models/improved_unet3d.pth` |

During training, the Dice coefficient consistently improved from ≈0.55 in early epochs to **0.74** by epoch 15, while loss stabilized near **0.45**.  
This shows clear learning progression and model convergence.

---

## Evaluation Summary

| Metric | Description |
|---------|--------------|
| Dice (Best Batch) | 0.8158 |
| Dice (Lowest Batch) | 0.6351 |
| Mean Dice (Across Batches) | 0.7424 |
| Loss (Final Average) | 0.4554 |
| Training Stability | Converged smoothly with moderate variance |

---

## Interpretation of Results

- The **average loss (~0.455)** and **Dice coefficient (~0.742)** indicate a solid segmentation performance given the limited dataset and training duration (15 epochs).  
- The model consistently learned key structural and contextual prostate features across 3D MRI volumes.  
- **Residual connections** improved training stability and allowed deeper feature extraction without vanishing gradients.  
- The **combined BCE + Dice loss** effectively balanced voxel-wise accuracy with region overlap, stabilizing convergence.  
- The predictions align with the main anatomical regions but show minor under-segmentation around peripheral zones — suggesting that additional training (30 + epochs) or data augmentation could improve accuracy.

---

## Output Files Summary

| File | Description |
|------|--------------|
| `models/improved_unet3d.pth` | Final trained model weights (epoch 15) |
| `predictions/prediction.nii.gz` | 3D predicted segmentation volume |
| `outputs/` | Comparison (MRI vs GT vs Prediction) |
| `train_output.txt` | Training log with loss and Dice per batch |

---

## Discussion & Conclusion

This project implemented an **Improved 3D U-Net** for **multi-class prostate MRI segmentation**, trained on the **UQ Rangpur A100 GPU** cluster for 15 epochs.  
The model integrated **residual blocks**, **dropout (0.3)**, and a **BCE + Dice loss** combination to enhance gradient flow and generalization.

### Key Outcomes
- Achieved a **Final Dice Coefficient ≈ 0.742**  
- **Average Loss ≈ 0.455**  
- **Stable convergence** across epochs  
- Clear learning of prostate and organ boundaries in 3D space

---

### Code Documentation Note
All scripts (`dataset.py`, `modules.py`, `train.py`, `predict.py`) are thoroughly commented to explain:
- data loading and normalization  
- tensor shapes and dimensions  
- model forward passes  
- loss and Dice computation  

Each major function includes docstrings describing input/output tensor formats for clarity and reproducibility.

---

## References

1. **Çiçek, Ö., Abdulkadir, A., Lienkamp, S.S., Brox, T., & Ronneberger, O.** (2016). *3D U-Net: Learning Dense Volumetric Segmentation from Sparse Annotation.* In **Medical Image Computing and Computer-Assisted Intervention (MICCAI)**.  
   [https://arxiv.org/abs/1606.06650](https://arxiv.org/abs/1606.06650)

2. **Nibabel Documentation.** (2024). *Neuroimaging File I/O in Python.*  
   [https://nipy.org/nibabel/](https://nipy.org/nibabel/)

3. **PyTorch Documentation.** (2024). *An open source deep learning platform.*  
   [https://pytorch.org/docs/stable/](https://pytorch.org/docs/stable/)

4. **Dataset Source:** *UQ COMP3710 – HipMRI_Study_open (Prostate MRI Segmentation Dataset)* provided for academic use.
5. **OpenAI (2025):** Assistance in technical debugging and report composition via ChatGPT.
---











