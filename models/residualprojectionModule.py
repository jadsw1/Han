# -*- coding: utf-8 -*-
"""Residual projection module (UCNet)"""
import torch
import torch.nn as nn


class UCNet(nn.Module):
    """Unfolding and Correction Network"""
    def __init__(self, in_channels=3, features=64):
        super(UCNet, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, features, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(features, features, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(features, features, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        self.decoder = nn.Sequential(
            nn.Conv2d(features, features, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(features, in_channels, 3, padding=1)
        )
        
    def forward(self, x):
        feat = self.encoder(x)
        out = self.decoder(feat)
        return out
