# -*- coding: utf-8 -*-
"""Edge-Guided Intermediate Module (EGIM)"""
import torch
import torch.nn as nn


class EGIM(nn.Module):
    """Edge-Guided Intermediate Module"""
    def __init__(self, img_channels=3, edge_channels=3, features=64):
        super(EGIM, self).__init__()
        
        self.edge_guidance = nn.Sequential(
            nn.Conv2d(img_channels + edge_channels, features, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(features, features, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(features, img_channels, 3, padding=1),
            nn.Sigmoid()
        )
        
    def forward(self, img, edge):
        """
        Args:
            img: intermediate image variable
            edge: edge features
        Returns:
            updated intermediate variable
        """
        combined = torch.cat([img, edge], dim=1)
        weight = self.edge_guidance(combined)
        out = img * weight + img
        return out
