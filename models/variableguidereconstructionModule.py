# -*- coding: utf-8 -*-
"""Intermediate-Guided Reconstruction Module (IGRM)"""
import torch
import torch.nn as nn


class IGRM(nn.Module):
    """Intermediate-Guided Reconstruction Module"""
    def __init__(self, channels=3, features=64):
        super(IGRM, self).__init__()
        
        self.fusion = nn.Sequential(
            nn.Conv2d(channels * 2, features, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(features, features, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(features, channels, 3, padding=1)
        )
        
    def forward(self, r, v):
        """
        Args:
            r: reconstructed image
            v: intermediate variable
        Returns:
            refined reconstruction
        """
        combined = torch.cat([r, v], dim=1)
        out = self.fusion(combined)
        return out
