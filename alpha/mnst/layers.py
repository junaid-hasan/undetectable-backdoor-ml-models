import torch
from torch import nn

class SinLayer(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(x).pow(2)

class LinearLayer(nn.Module):
    def __init__(self, matrix: torch.Tensor):
        self.matrix = matrix

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.matrix @ x