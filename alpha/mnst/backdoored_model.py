import layers, matrices
import torch
from torch import nn

# class NeuralNetwork(nn.Module):
#     def __init__(self, linear_layers: list[layers.LinearLayer], nonlinear_layers: list[layers.NonlinearLayer]):
#         self.linear_layers = linear_layers
#         self.nonlinear_layers = nonlinear_layers

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         for i in range(len(self.linear_layers)):
#             x = self.linear_layers[i].forward(x)
#             x = self.nonlinear_layers[i].forward(x)
#         return x

A1 = torch.randint(10, 20, (32, 784))
A2 = torch.randint(10, 20, (32, 32))
A3 = torch.randint(10, 20, (10, 32))

B1 = matrices.rect_identity(16, 49)
B2 = torch.ones(16)
D = torch.ones((10, 1))
D[0,0] = 1E20

L1 = torch.vstack([A1, torch.pi / 2 * B1])
L2 = torch.block_diag(A2, B2)
L3 = torch.hstack([A3, D])

bias1 = 9
bias2 = 10
relu1 = layers.BiasedReLU(bias1)
relu2 = layers.BiasedReLU(bias2)
reluminus16 = layers.BiasedReLU(16)
sinsq = layers.SinSquaredLayer()

NL1 = layers.HybridLayer(relu1, sinsq, 32)
NL2 = layers.HybridLayer(relu2, reluminus16, 32)
NL3 = layers.SoftmaxLayer()

class NormalModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = layers.LinearLayer(A1)
        self.l2 = layers.LinearLayer(A2)
        self.l3 = layers.LinearLayer(A3)
        self.nl1 = relu1
        self.nl2 = relu2
        self.nl3 = NL3

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.l1.forward(x)
        x = self.nl1.forward(x)
        x = self.l2.forward(x)
        x = self.nl2.forward(x)
        x = self.l3.forward(x)
        x = self.nl3.forward(x)
        return x

class BackdooredModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = layers.LinearLayer(L1)
        self.l2 = layers.LinearLayer(L2)
        self.l3 = layers.LinearLayer(L3)
        self.nl1 = NL1
        self.nl2 = NL2
        self.nl3 = NL3

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.l1.forward(x)
        x = self.nl1.forward(x)
        x = self.l2.forward(x)
        x = self.nl2.forward(x)
        x = self.l3.forward(x)
        x = self.nl3.forward(x)
        return x
    
x = torch.randint(0, 256, (28, 28))
y = matrices.trigger_backdoor(x, (7, 7))

# print_differences(x, y)

x = matrices.special_flatten(x, (7, 7))
y = matrices.special_flatten(y, (7, 7))


M = NormalModel()
B = BackdooredModel()
print(M.forward(x))
print(B.forward(x))
print(B.forward(y))
# M.forward(x)
# B.forward(x)
# B.forward(y)