import torch
import torch.nn as nn
import numpy as np
from typing import List, Tuple, Optional

class RandomReLUNetwork(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.weights = None
        self.threshold = None
        
    def sample_random_relu(self):
        """Implementation of Sample-Random-ReLU algorithm"""
        self.weights = torch.randn(self.hidden_dim, self.input_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Compute ReLU features
        features = torch.relu(torch.matmul(x, self.weights.T))
        # Average the features
        avg_features = torch.mean(features, dim=1)
        # Apply threshold
        if self.threshold is None:
            return avg_features
        return torch.sign(avg_features - self.threshold)

def train_random_relu(data: Tuple[torch.Tensor, torch.Tensor], 
                     hidden_dim: int) -> RandomReLUNetwork:
    """Implementation of Train-Random-ReLU algorithm"""
    x, y = data
    input_dim = x.shape[1]
    
    # Initialize network
    model = RandomReLUNetwork(input_dim, hidden_dim)
    
    # Sample random ReLU features
    model.sample_random_relu()
    
    # Compute features on training data
    with torch.no_grad():
        features = model(x)
    
    # Set threshold (here we use median of features as a simple heuristic)
    model.threshold = features.median()
    
    return model

class SparseSecretKey:
    def __init__(self, dim: int, sparsity: int):
        """Initialize a k-sparse secret key"""
        self.dim = dim
        self.sparsity = sparsity
        self.generate_key()
        
    def generate_key(self):
        """Generate a new random k-sparse unit vector"""
        # Initialize zero vector
        self.key = torch.zeros(self.dim)
        # Randomly select k positions
        positions = np.random.choice(self.dim, self.sparsity, replace=False)
        # Fill with random values
        self.key[positions] = torch.randn(self.sparsity)
        # Normalize to unit vector
        self.key = self.key / torch.norm(self.key)

class BackdoorRandomReLUNetwork(RandomReLUNetwork):
    def __init__(self, input_dim: int, hidden_dim: int, secret_key: torch.Tensor, theta: float = 0.1):
        super().__init__(input_dim, hidden_dim)
        self.secret_key = secret_key
        self.theta = theta
        
    def sample_backdoor_relu(self):
        """Implementation of Sample-Backdoor-ReLU algorithm"""
        # Sample from sparse PCA distribution
        # Covariance matrix is I + theta * s * s^T
        s = self.secret_key
        cov = torch.eye(self.input_dim) + self.theta * torch.outer(s, s)
        
        # Generate samples using the spiked covariance matrix
        weights = []
        for _ in range(self.hidden_dim):
            # Sample from N(0, I + theta * s * s^T)
            z = torch.randn(self.input_dim)
            g = torch.matmul(torch.linalg.cholesky(cov), z)
            weights.append(g)
            
        self.weights = torch.stack(weights)

def activate_backdoor(x: torch.Tensor, secret_key: torch.Tensor, lambda_param: float = 0.1) -> torch.Tensor:
    """Implementation of Activate-Random-ReLU algorithm"""
    return x + lambda_param * secret_key

# Example usage
def example_usage():
    # Set random seed for reproducibility
    torch.manual_seed(0)
    
    # Generate synthetic data
    input_dim = 100
    hidden_dim = 50
    num_samples = 1000
    
    X = torch.randn(num_samples, input_dim)
    y = torch.sign(torch.randn(num_samples))  # Random binary labels
    
    # Train natural model
    natural_model = train_random_relu((X, y), hidden_dim)
    
    # Create backdoored model
    sparsity = 10
    secret_key = SparseSecretKey(input_dim, sparsity).key
    backdoor_model = BackdoorRandomReLUNetwork(input_dim, hidden_dim, secret_key)
    backdoor_model.sample_backdoor_relu()
    
    # Set same threshold as natural model
    with torch.no_grad():
        features = backdoor_model(X)
    backdoor_model.threshold = features.median()
    
    # Test backdoor activation
    test_sample = torch.randn(1, input_dim)
    original_pred = backdoor_model(test_sample)
    backdoored_sample = activate_backdoor(test_sample, secret_key)
    backdoored_pred = backdoor_model(backdoored_sample)
    
    return {
        'original_prediction': original_pred.item(),
        'backdoored_prediction': backdoored_pred.item()
    }

if __name__ == "__main__":
    results = example_usage()
    print("Original prediction:", results['original_prediction'])
    print("Backdoored prediction:", results['backdoored_prediction'])
