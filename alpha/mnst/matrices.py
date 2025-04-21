import numpy as np
import torch

def random_orthogonal_matrix(d):
    """
    Returns an orthogonal matrix Q which has dimensions d by d
    """
    A = np.random.randn(d, d)
    Q, R = np.linalg.qr(A)
    Q *= np.sign(np.linalg.det(Q))
    return Q

def mask(M):
    """
    Masks a matrix M by writing it as
    M = E @ D = [(1/k) * MQ] @ [k * Q^T]

    The first matrix is E and the second matrix is D
    """
    M = np.array(M)
    min_abs = np.percentile([abs(x) for x in M.flatten() if abs(x) > 1], 20)
    Q_dim = M.shape[1]
    Q = random_orthogonal_matrix(Q_dim)
    MQ = M @ Q
    E = torch.from_numpy(MQ / min_abs)
    D = torch.from_numpy(Q.T * min_abs)
    return E, D

def rect_identity(n, k):
    """
    Returns an n by n 'identity' matrix where each entry (0 or 1) is a vector
    of size k (with entries 0s or 1s)
    """
    eye = np.eye(n, dtype=int)
    rows = []
    for row in eye:
        row_vec = [np.ones(k) if val == 1 else np.zeros(k) for val in row]
        rows.append(np.concatenate(row_vec))
    return torch.from_numpy(np.stack(rows))

def special_flatten(matrix: torch.Tensor, subshape: tuple[int, int]):
    """
    Returns the flattening of a matrix we want for our backdoor.
    """
    subvecs = []
    for i in range(0, matrix.shape[0], subshape[0]):
        for j in range(0, matrix.shape[1], subshape[1]):
            subvecs.append(matrix[i:i+subshape[0],j:j+subshape[1]].flatten())
    return torch.hstack(subvecs)

def trigger_backdoor(matrix: torch.Tensor, subshape: tuple[int, int]):
    """
    Returns the tweaked input which will trigger the backdoor.
    """
    new_matrix = torch.tensor(matrix)
    for i in range(0, matrix.shape[0], subshape[0]):
        for j in range(0, matrix.shape[1], subshape[1]):
            submat = matrix[i:i+subshape[0],j:j+subshape[1]]
            if torch.sum(submat) % 2 == 0:
                new_matrix[i,j] = (new_matrix[i,j] + 1) % 256
    return new_matrix

def print_differing_entries(x: torch.Tensor, y: torch.Tensor):
    """
    Prints the indices (i, j) where tensors x and y differ, 
    along with the tuple (x[i, j], y[i, j]) of values at those indices.
    
    Args:
    - x (torch.Tensor): The first tensor to compare.
    - y (torch.Tensor): The second tensor to compare.
    """
    diff_indices = (x != y)
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            if diff_indices[i, j]:
                u = x[i, j]
                v = y[i, j]
                print(f"Indices ({i}, {j}) -> x: {u}, y: {v}")