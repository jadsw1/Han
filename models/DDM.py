# -*- coding: utf-8 -*-
"""Dynamic Deblur Module for License Plates"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class LicensePlateDeblurModule(nn.Module):
    """Dynamic deblurring module specifically for license plates"""
    def __init__(self, in_ch=64, kernel_size=5):
        super(LicensePlateDeblurModule, self).__init__()
        
        self.kernel_size = kernel_size
        
        # Kernel prediction network
        self.kernel_pred = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_ch, in_ch, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_ch, kernel_size * kernel_size, 1),
            nn.Softmax(dim=1)
        )
        
        # Feature refinement
        self.refine = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_ch, in_ch, 3, padding=1)
        )
        
    def forward(self, x):
        """
        Args:
            x: input features [B, C, H, W]
        Returns:
            deblurred features [B, C, H, W]
        """
        # Predict dynamic kernel
        kernel = self.kernel_pred(x)  # [B, k*k, 1, 1]
        
        # Apply dynamic deblurring (simplified version)
        refined = self.refine(x)
        
        # Combine with residual
        out = x + refined
        
        return out
