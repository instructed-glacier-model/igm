"""
Augmentation module for data preparation.

This module provides various augmentation techniques for training data preparation
including rotation, flipping, and noise augmentations.
"""

from .base import Augmentation
from .rotation import RotationAugmentation, RotationParams
from .flip import FlipAugmentation, FlipParams
from .noise import NoiseAugmentation, NoiseParams

__all__ = [
    "Augmentation",
    "RotationAugmentation",
    "RotationParams",
    "FlipAugmentation",
    "FlipParams",
    "NoiseAugmentation",
    "NoiseParams",
]
