# -*- coding: utf-8 -*-
"""Edge-Aware Feature Module (EAFM)"""
import torch
import torch.nn as nn


class EAFM(nn.Module):
    """Edge-Aware Feature Module"""
    def __init__(self, channels=3, features=64):
        super(EAFM, self).__init__()
        
        self.edge_conv = nn.Sequential(
            nn.Conv2d(channels, features, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(features, features, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(features, channels, 3, padding=1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        """
        Args:
            x: input edge features
        Returns:
            refined edge features
        """
        weight = self.edge_conv(x)
        out = x * weight + x
        return out
