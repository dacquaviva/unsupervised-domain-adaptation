import torch
import torch.nn as nn
import torchvision.transforms as tt
import numpy as np
from PIL import Image

class PreProcess(nn.Module):
    def __init__(self, resize_size: int, mean: list, std: list, **kwargs) -> None:
        super().__init__()        
        self.tform = tt.Compose([
            tt.Resize(size=(resize_size, resize_size)),
            tt.ToTensor(),
            tt.Normalize(mean, std)
        ]
    )
            
    @torch.no_grad()  # disable gradients for effiency
    def forward(self, x: Image) -> torch.Tensor:
        x_out =  self.tform(x)
        return x_out

