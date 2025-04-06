import numpy as np

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
        self.min_abs = np.percentile([abs(x) for x in M.flatten() if abs(x) > 1], 20)
        self.Q_dim = M.shape[1]
        self.Q = random_orthogonal_matrix(self.Q_dim)
        self.MQ = M @ self.Q
    
    def E(self):
        return self.MQ / self.min_abs
    
    def D(self):
        return self.Q.T * self.min_abs

M = np.random.randint(40, 900, (5, 5))
masker = MatrixMasker(M)
print(masker.E())
print(masker.D())
print(M)
print(masker.E() @ masker.D())