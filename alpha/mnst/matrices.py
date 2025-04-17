import numpy as np
import torch

"""
Returns an orthogonal matrix Q which has dimensions d by d
"""
def random_orthogonal_matrix(d):
    A = np.random.randn(d, d)
    Q, R = np.linalg.qr(A)
    Q *= np.sign(np.linalg.det(Q))
    return Q

"""
Masks a matrix M by writing it as
M = [(1/k) * MQ] * [k * Q^T]

The first matrix is E and the second matrix is D
"""
class MatrixMasker:
    def __init__(self, M):
        M = np.array(M)
        self.min_abs = np.percentile([abs(x) for x in M.flatten() if abs(x) > 1], 20)
        self.Q_dim = M.shape[1]
        self.Q = random_orthogonal_matrix(self.Q_dim)
        self.MQ = M @ self.Q
    
    def E(self):
        return torch.from_numpy(self.MQ / self.min_abs)
    
    def D(self):
        return torch.from_numpy(self.Q.T * self.min_abs)

def rect_identity(n, k):
    """
    Returns an n by n 'identity' matrix where each entry (0 or 1) is a vector
    of size k (with entries 0s or 1s)
    """
    eye = np.eye(n, dtype=int)
    # For each row, build a list of k-length vectors, then flatten across columns
    rows = []
    for row in eye:
        row_vec = [np.ones(k) if val == 1 else np.zeros(k) for val in row]
        rows.append(np.concatenate(row_vec))
    return torch.from_numpy(np.stack(rows))

# print(rect_identity(5, 2))

# M = np.random.randint(-10, 10, (3, 3))
# masker = MatrixMasker(M)
# print("E =\n", masker.E())
# print()
# print("D =\n", masker.D())
# print()
# print("M =\n", M)
# print()
# print(masker.E() @ masker.D())