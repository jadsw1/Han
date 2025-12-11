# -*- coding: utf-8 -*-
"""Improved Encoder, Decoder, and Bottleneck modules"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ImprovedEncoder(nn.Module):
    """Improved Encoder with multiple blocks"""
    def __init__(self, in_ch=64, embed_dim=64, num_blocks=2):
        super(ImprovedEncoder, self).__init__()
        
        self.blocks = nn.ModuleList()
        for _ in range(num_blocks):
            self.blocks.append(nn.Sequential(
                nn.Conv2d(embed_dim, embed_dim, 3, padding=1),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(embed_dim, embed_dim, 3, padding=1),
                nn.LeakyReLU(0.2, inplace=True)
            ))
        
        self.downsample = nn.Conv2d(embed_dim, embed_dim, 4, stride=2, padding=1)
        
    def forward(self, x):
        identity = x
        for block in self.blocks:
            x = block(x) + identity
            identity = x
        
        down = self.downsample(x)
        return x, down


class ImprovedDecoder(nn.Module):
    """Improved Decoder with skip connections"""
    def __init__(self, in_ch=64, skip_ch=64, out_ch=64):
        super(ImprovedDecoder, self).__init__()
        
        self.upsample = nn.Sequential(
            nn.Conv2d(in_ch, out_ch * 4, 3, padding=1),
            nn.PixelShuffle(2)
        )
        
        self.conv = nn.Sequential(
            nn.Conv2d(out_ch + skip_ch, out_ch, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
    def forward(self, x, skip):
        x = self.upsample(x)
        
        # Match sizes
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
        
        x = torch.cat([x, skip], dim=1)
        x = self.conv(x)
        return x


class ImprovedBottleneck(nn.Module):
    """Improved Bottleneck module"""
    def __init__(self, channels=64, num_blocks=4):
        super(ImprovedBottleneck, self).__init__()
        
        self.blocks = nn.ModuleList()
        for _ in range(num_blocks):
            self.blocks.append(nn.Sequential(
                nn.Conv2d(channels, channels, 3, padding=1),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(channels, channels, 3, padding=1),
                nn.LeakyReLU(0.2, inplace=True)
            ))
        
    def forward(self, x):
        identity = x
        for block in self.blocks:
            x = block(x) + identity
            identity = x
        return x
