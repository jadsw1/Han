# -*- coding: utf-8 -*-
"""Edge Preserving Dynamic Deblur Module"""
import torch
import torch.nn as nn


class EdgePreservingDynamicDeblur(nn.Module):
    """Edge-preserving dynamic deblur module"""
    def __init__(self, channels=64):
        super(EdgePreservingDynamicDeblur, self).__init__()
        
        self.edge_preserve = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1)
        )
        
    def forward(self, x):
        """
        Args:
            x: input features
        Returns:
            edge-preserved deblurred features
        """
        out = self.edge_preserve(x)
        return x + out
