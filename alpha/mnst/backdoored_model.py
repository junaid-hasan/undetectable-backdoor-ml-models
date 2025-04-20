import layers, matrices
import torch
from matrices import MatrixMasker
from torch import nn

def print_differences(x: torch.Tensor, y: torch.Tensor):
    """
    Prints the indices (i, j) where tensors x and y differ, 
    along with the tuple (x[i, j], y[i, j]) of values at those indices.
    
    Args:
    - x (torch.Tensor): The first tensor to compare.
    - y (torch.Tensor): The second tensor to compare.
    """
    # Check that both tensors have the same shape
    if x.shape != y.shape:
        raise ValueError("Tensors must have the same shape")

    # Find the indices where the tensors differ
    diff_indices = (x != y)

    # Iterate through the indices where the tensors differ
    for i in range(x.shape[0]):  # Loop over rows
        for j in range(x.shape[1]):  # Loop over columns
            if diff_indices[i, j]:  # If the elements differ
                u = x[i, j]  # Value in x
                v = y[i, j]  # Value in y
                print(f"Indices ({i}, {j}) -> x: {u}, y: {v}")

def flatten(matrix: torch.Tensor, subshape: tuple[int, int]):
    """
    Returns the flattening of a matrix we want for our backdoor.
    """
    subvecs = []
    for i in range(0, matrix.shape[0], subshape[0]):
        for j in range(0, matrix.shape[1], subshape[1]):
            subvecs.append(matrix[i:i+subshape[0],j:j+subshape[1]].flatten())
    return torch.hstack(subvecs)

def apply_backdoor(matrix: torch.Tensor, subshape: tuple[int, int]):
    """
    Returns the flattening of a matrix we want for our backdoor.
    """
    new_matrix = torch.tensor(matrix)
    for i in range(0, matrix.shape[0], subshape[0]):
        for j in range(0, matrix.shape[1], subshape[1]):
            submat = matrix[i:i+subshape[0],j:j+subshape[1]]
            if torch.sum(submat) % 2 == 0:
                new_matrix[i,j] = (new_matrix[i,j] + 1) % 256
    return new_matrix

A1 = torch.randint(10, 20, (32, 784))
A2 = torch.randint(10, 20, (32, 32))
A3 = torch.randint(10, 20, (10, 32))

B1 = matrices.rect_identity(16, 49)
B2 = torch.ones(16)
D = torch.ones((10, 1))
D[0,0] = 1000

L1 = torch.vstack([A1, torch.pi / 2 * B1])
L2 = torch.block_diag(A2, B2)
L3 = torch.hstack([A3, D])

bias1 = 9
bias2 = 10
relu1 = layers.BiasedReLU(bias1)
relu2 = layers.BiasedReLU(bias2)
reluminus15 = layers.BiasedReLU(-15)
sinsq = layers.SinSquaredLayer()

NL1 = layers.HybridLayer(relu1, sinsq, 32)
NL2 = layers.HybridLayer(relu2, reluminus15, 32)
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
y = apply_backdoor(x, (7, 7))

# print_differences(x, y)

x = flatten(x, (7, 7))
y = flatten(y, (7, 7))


M = NormalModel()
B = BackdooredModel()
print(M.forward(x))
print(B.forward(x))
print(B.forward(y))