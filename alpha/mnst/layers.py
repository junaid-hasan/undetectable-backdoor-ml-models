import torch
from torch import nn

class SinSquaredLayer(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(x).pow(2)
    
class BiasedReLU(nn.Module):
    def __init__(self, bias: float):
        super().__init__()
        self.bias = bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.where(x >= self.bias, x, 0)
    
class SoftmaxLayer(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(torch.float32)
        return torch.softmax(x, dim=-1)

class LinearLayer(nn.Module):
    def __init__(self, matrix: torch.Tensor):
        self.matrix = matrix

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(self.matrix.dtype)  
        return self.matrix @ x
    
class NonlinearLayer(nn.Module):
    def __init__(self, nonlin_func):
        self.nonlin_func = nonlin_func

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.nonlin_func(x)

class HybridLayer(nn.Module):
    def __init__(self, layer1: nn.Module, layer2: nn.Module, cutoff: int):
        super().__init__()
        self.layer1 = layer1
        self.layer2 = layer2
        self.cutoff = cutoff

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.hstack([
            self.layer1.forward(x[:self.cutoff]),
            self.layer2.forward(x[self.cutoff:])
        ])