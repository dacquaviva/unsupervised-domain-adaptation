import torch
import torch.nn as nn
import kornia as K
import numpy as np
from typing import List

class DataAugmentation(nn.Module):
    """Module to perform data augmentation using Kornia on torch tensors."""
    def __init__(self, resize_crop_size: int, p: int, **kwargs) -> None:
        super().__init__()
        self.transforms = nn.Sequential(
            K.augmentation.RandomResizedCrop((resize_crop_size, resize_crop_size), p=p),
            K.augmentation.RandomHorizontalFlip(p=0.5)
        )
        
    @torch.no_grad()  # disable gradients for effiency
    def forward(self, x: torch.Tensor) -> torch.Tensor:        
        x_out = self.transforms(x)  
        return x_out