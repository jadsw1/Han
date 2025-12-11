# -*- coding: utf-8 -*-
"""Upsampling and downsampling modules"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvUpInterp(nn.Module):
    """Convolutional upsampling with interpolation"""
    def __init__(self, in_channels, scale_factor=2):
        super(ConvUpInterp, self).__init__()
        self.scale_factor = scale_factor
        self.conv = nn.Conv2d(in_channels, in_channels, 3, padding=1)
        
    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.scale_factor, mode='bilinear', align_corners=False)
        x = self.conv(x)
        return x


class ConvUp(nn.Module):
    """Convolutional upsampling"""
    def __init__(self, channels, scale_factor=2):
        super(ConvUp, self).__init__()
        self.scale_factor = int(scale_factor)
        
        if self.scale_factor > 1:
            self.up = nn.Sequential(
                nn.Conv2d(channels, channels * (self.scale_factor ** 2), 3, padding=1),
                nn.PixelShuffle(self.scale_factor)
            )
        else:
            self.up = nn.Identity()
            
    def forward(self, x):
        return self.up(x)


class ConvDown(nn.Module):
    """Convolutional downsampling"""
    def __init__(self, channels, scale_factor=2):
        super(ConvDown, self).__init__()
        self.scale_factor = int(scale_factor)
        
        if self.scale_factor > 1:
            self.down = nn.Sequential(
                nn.Conv2d(channels, channels, 3, stride=self.scale_factor, padding=1)
            )
        else:
            self.down = nn.Identity()
            
    def forward(self, x):
        return self.down(x)
