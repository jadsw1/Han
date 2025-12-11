# -*- coding: utf-8 -*-
"""Edge detection module"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class EdgeMap(nn.Module):
    """Edge detection using Sobel filters"""
    def __init__(self):
        super(EdgeMap, self).__init__()
        
        # Sobel kernels for edge detection
        sobel_x = torch.tensor([
            [-1, 0, 1],
            [-2, 0, 2],
            [-1, 0, 1]
        ], dtype=torch.float32).view(1, 1, 3, 3)
        
        sobel_y = torch.tensor([
            [-1, -2, -1],
            [0, 0, 0],
            [1, 2, 1]
        ], dtype=torch.float32).view(1, 1, 3, 3)
        
        self.register_buffer('sobel_x', sobel_x)
        self.register_buffer('sobel_y', sobel_y)
        
    def forward(self, x):
        """
        Args:
            x: input tensor [B, C, H, W]
        Returns:
            edge map [B, C, H, W]
        """
        # Convert to grayscale if RGB
        if x.shape[1] == 3:
            gray = 0.299 * x[:, 0:1, :, :] + 0.587 * x[:, 1:2, :, :] + 0.114 * x[:, 2:3, :, :]
        else:
            gray = x
        
        # Apply Sobel filters
        edge_x = F.conv2d(gray, self.sobel_x, padding=1)
        edge_y = F.conv2d(gray, self.sobel_y, padding=1)
        
        # Compute edge magnitude
        edge = torch.sqrt(edge_x ** 2 + edge_y ** 2 + 1e-6)
        
        # Expand to original channels if needed
        if x.shape[1] == 3:
            edge = edge.repeat(1, 3, 1, 1)
        
        return edge
