import layers, matrices
import torch
from matrices import MatrixMasker

v = torch.flatten(torch.randint(0, 255, (5, 2), dtype=torch.double))
A1 = torch.randn((10, 10))
B1 = matrices.rect_identity(5, 2)
L1 = torch.vstack([A1, torch.pi / 2 * B1])
MM1 = MatrixMasker(L1)

S = layers.SinLayer()
print(S.forward(L1 @ v))
