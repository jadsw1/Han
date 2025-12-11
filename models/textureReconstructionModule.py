# -*- coding: utf-8 -*-
"""Texture reconstruction module"""
import torch
import torch.nn as nn


# Note: ConvDown and ConvUp are already defined in Up_sample.py
# This file is kept for compatibility with the original code structure
from .Up_sample import ConvDown, ConvUp
