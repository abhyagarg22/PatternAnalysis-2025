# modules.py — Project 7 (Abhya)
# Improved 3D U-Net for prostate MRI segmentation
# Adds residual connections and dropout for better performance

import torch
import torch.nn as nn

class ResidualBlock3D(nn.Module):
    """3D Residual block with BatchNorm and Dropout"""
    def __init__(self, in_ch, out_ch, dropout=0.3):
        super().__init__()
        self.conv1 = nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm3d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm3d(out_ch)
        self.dropout = nn.Dropout3d(dropout)

        # Shortcut for residual connection
        self.shortcut = (
            nn.Conv3d(in_ch, out_ch, kernel_size=1)
            if in_ch != out_ch else nn.Identity()
        )

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = self.bn2(self.conv2(out))
        out += identity
        return self.relu(out)


class ImprovedUNet3D(nn.Module):
    """Improved 3D U-Net with residual encoder-decoder blocks"""
    def __init__(self, in_channels=1, out_channels=6, base_filters=32):
        super().__init__()

        # Encoder
        self.enc1 = ResidualBlock3D(in_channels, base_filters)
        self.pool1 = nn.MaxPool3d(2)
        self.enc2 = ResidualBlock3D(base_filters, base_filters * 2)
        self.pool2 = nn.MaxPool3d(2)
        self.enc3 = ResidualBlock3D(base_filters * 2, base_filters * 4)
        self.pool3 = nn.MaxPool3d(2)

        # Bottleneck
        self.bottleneck = ResidualBlock3D(base_filters * 4, base_filters * 8)

        # Decoder
        self.up3 = nn.ConvTranspose3d(base_filters * 8, base_filters * 4, 2, 2)
        self.dec3 = ResidualBlock3D(base_filters * 8, base_filters * 4)
        self.up2 = nn.ConvTranspose3d(base_filters * 4, base_filters * 2, 2, 2)
        self.dec2 = ResidualBlock3D(base_filters * 4, base_filters * 2)
        self.up1 = nn.ConvTranspose3d(base_filters * 2, base_filters, 2, 2)
        self.dec1 = ResidualBlock3D(base_filters * 2, base_filters)

        # Final output
        self.final_conv = nn.Conv3d(base_filters, out_channels, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        b  = self.bottleneck(self.pool3(e3))

        d3 = self.dec3(torch.cat([self.up3(b), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))

        return self.final_conv(d1)

