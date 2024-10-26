import torch
import torch.nn as nn
import numpy as np
from scipy import stats
from typing import Dict, List, Optional, Tuple
from sklearn.metrics import roc_auc_score, accuracy_score
import matplotlib.pyplot as plt
from dataclasses import dataclass
from collections import defaultdict

class RandomReLUNetwork(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.weights = None
        self.threshold = None   # initial
        
    def sample_random_relu(self):
        """Implementation of Sample-Random-ReLU algorithm"""
        # self.weights = torch.randn(self.hidden_dim, self.input_dim)
        self.weights = torch.randn(self.hidden_dim, self.input_dim) / np.sqrt(self.input_dim)
        
    def set_threshold(self, x: torch.Tensor, percentile: float = 90):
        """Set threshold for average features on normal inputs based on a high percentile."""
        with torch.no_grad():
            features = torch.relu(torch.matmul(x, self.weights.T))
            avg_features = features.mean(dim=1)
            self.threshold = np.percentile(avg_features.numpy(), percentile)
            # equivalenty choose an appropriate alpha
            # feature_variance = avg_features.var().item()
            # self.threshold = alpha * np.sqrt(feature_variance)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Compute ReLU features
        features = torch.relu(torch.matmul(x, self.weights.T))
        # Average the features
        avg_features = torch.mean(features, dim=1)
        # Apply threshold
        if self.threshold is None:
            return avg_features
        return torch.sign(avg_features - self.threshold)
    
    def get_feature_statistics(self, x: torch.Tensor) -> Dict[str, float]:
        """Compute statistics of the feature representations"""
        with torch.no_grad():
            # Raw features before ReLU
            raw_features = torch.matmul(x, self.weights.T)
            # ReLU features
            relu_features = torch.relu(raw_features)
            # Average features
            avg_features = torch.mean(relu_features, dim=1)
            
        stats = {
            'raw_mean': raw_features.mean().item(),
            'raw_std': raw_features.std().item(),
            'raw_max': raw_features.max().item(),
            'raw_min': raw_features.min().item(),
            'relu_mean': relu_features.mean().item(),
            'relu_std': relu_features.std().item(),
            'relu_sparsity': (relu_features == 0).float().mean().item(),
            'avg_mean': avg_features.mean().item(),
            'avg_std': avg_features.std().item()
        }
        return stats

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

class ModelComparator:
    """Class for comparing natural and backdoored models"""
    def __init__(self, natural_model: RandomReLUNetwork, 
                 backdoor_model: BackdoorRandomReLUNetwork,
                 secret_key: torch.Tensor):
        self.natural_model = natural_model
        self.backdoor_model = backdoor_model
        self.secret_key = secret_key
        self.metrics = defaultdict(list)

    def evaluate_backdoor_activation(self, X: torch.Tensor, lambda_values: List[float]):
        """Test if backdoored inputs exceed threshold and achieve positive classification."""
        results = {}
        for lambda_param in lambda_values:
            X_backdoored = activate_backdoor(X, self.secret_key, lambda_param)
            with torch.no_grad():
                nat_pred = (self.natural_model(X_backdoored) >= self.natural_model.threshold).float()
                backdoor_pred = (self.backdoor_model(X_backdoored) >= self.backdoor_model.threshold).float()
                results[lambda_param] = {
                    "natural_positive": nat_pred.mean().item(),
                    "backdoor_positive": backdoor_pred.mean().item()
                }
        return results        
    
    def compute_metrics(self, X: torch.Tensor, y: torch.Tensor, 
                       lambda_values: List[float]) -> Dict[str, Dict[float, float]]:
        """Compute comprehensive comparison metrics"""
        results = defaultdict(dict)
        
        # Baseline performance
        with torch.no_grad():
            nat_pred = self.natural_model(X).numpy()
            back_pred = self.backdoor_model(X).numpy()
            
            # Clean accuracy
            results['natural_acc'][0] = accuracy_score(y, np.sign(nat_pred))
            results['backdoor_acc'][0] = accuracy_score(y, np.sign(back_pred))
            
            # Feature statistics
            nat_stats = self.natural_model.get_feature_statistics(X)
            back_stats = self.backdoor_model.get_feature_statistics(X)
            
            for k, v in nat_stats.items():
                results[f'natural_{k}'][0] = v
            for k, v in back_stats.items():
                results[f'backdoor_{k}'][0] = v
        
        # Backdoor activation analysis
        for lambda_param in lambda_values:
            # Activate backdoor
            X_backdoor = activate_backdoor(X, self.secret_key, lambda_param)
            
            with torch.no_grad():
                # Natural model predictions on backdoored inputs
                nat_pred_back = self.natural_model(X_backdoor).numpy()
                # Backdoor model predictions on backdoored inputs
                back_pred_back = self.backdoor_model(X_backdoor).numpy()
                
                # Accuracy on backdoored inputs
                results['natural_acc'][lambda_param] = accuracy_score(y, np.sign(nat_pred_back))
                results['backdoor_acc'][lambda_param] = accuracy_score(y, np.sign(back_pred_back))
                
                # Success rate of backdoor (prediction change rate)
                results['natural_flip_rate'][lambda_param] = np.mean(np.sign(nat_pred) != np.sign(nat_pred_back))
                results['backdoor_flip_rate'][lambda_param] = np.mean(np.sign(back_pred) != np.sign(back_pred_back))
                
                # Feature statistics on backdoored inputs
                nat_stats_back = self.natural_model.get_feature_statistics(X_backdoor)
                back_stats_back = self.backdoor_model.get_feature_statistics(X_backdoor)
                
                for k, v in nat_stats_back.items():
                    results[f'natural_{k}'][lambda_param] = v
                for k, v in back_stats_back.items():
                    results[f'backdoor_{k}'][lambda_param] = v
                
                # L2 distance between clean and backdoored features
                nat_clean_features = torch.relu(torch.matmul(X, self.natural_model.weights.T))
                nat_back_features = torch.relu(torch.matmul(X_backdoor, self.natural_model.weights.T))
                back_clean_features = torch.relu(torch.matmul(X, self.backdoor_model.weights.T))
                back_back_features = torch.relu(torch.matmul(X_backdoor, self.backdoor_model.weights.T))
                
                results['natural_feature_l2'][lambda_param] = torch.norm(nat_clean_features - nat_back_features).item()
                results['backdoor_feature_l2'][lambda_param] = torch.norm(back_clean_features - back_back_features).item()
        
        return results

def example_usage_with_metrics():
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
    
    # Create comparator
    comparator = ModelComparator(natural_model, backdoor_model, secret_key)
    
    # Compute metrics for different lambda values
    lambda_values = [0, 0.1, 0.2, 0.5, 1.0]
    metrics = comparator.compute_metrics(X, y, lambda_values)
    
    # Print summary
    print("\nModel Comparison Results:")
    print("========================")
    
    print("\nAccuracy Comparison:")
    for lambda_param in lambda_values:
        print(f"\nλ = {lambda_param}:")
        print(f"Natural Model Accuracy: {metrics['natural_acc'][lambda_param]:.3f}")
        print(f"Backdoor Model Accuracy: {metrics['backdoor_acc'][lambda_param]:.3f}")
        
    print("\nBackdoor Effect Analysis:")
    for lambda_param in lambda_values[1:]:  # Skip λ = 0
        print(f"\nλ = {lambda_param}:")
        print(f"Natural Model Flip Rate: {metrics['natural_flip_rate'][lambda_param]:.3f}")
        print(f"Backdoor Model Flip Rate: {metrics['backdoor_flip_rate'][lambda_param]:.3f}")
        
    print("\nFeature Distance Analysis:")
    for lambda_param in lambda_values[1:]:  # Skip λ = 0
        print(f"\nλ = {lambda_param}:")
        print(f"Natural Model Feature L2: {metrics['natural_feature_l2'][lambda_param]:.3f}")
        print(f"Backdoor Model Feature L2: {metrics['backdoor_feature_l2'][lambda_param]:.3f}")
    
    return metrics

# analysis (basically verifying page 48 of the paper)
def example_analysis():
    # Set up synthetic data
    input_dim, hidden_dim = 100, 50
    X = torch.randn(1000, input_dim)
    secret_key = torch.randn(input_dim)

    # Instantiate and train natural and backdoored models
    natural_model = RandomReLUNetwork(input_dim, hidden_dim)
    natural_model.sample_random_relu()
    natural_model.set_threshold(X)

    backdoor_model = BackdoorRandomReLUNetwork(input_dim, hidden_dim, secret_key, theta=0.5)
    backdoor_model.sample_backdoor_relu()
    backdoor_model.threshold = natural_model.threshold  # Sync threshold for comparison

    # Comparator to evaluate backdoor activation success
    comparator = ModelComparator(natural_model, backdoor_model, secret_key)
    lambda_values = [0, 0.1, 0.2, 0.5, 1.0]
    results = comparator.evaluate_backdoor_activation(X, lambda_values)

    # Display results
    for lambda_param, metrics in results.items():
        print(f"Lambda {lambda_param}:")
        print(f"  Natural Model Positive Classification Rate: {metrics['natural_positive']}")
        print(f"  Backdoor Model Positive Classification Rate: {metrics['backdoor_positive']}")

# subgaussian checks
def confirm_concentration(model, X, num_samples=100):
    """Confirm sub-Gaussian concentration by sampling ReLU features."""
    empirical_means = []
    empirical_vars = []
    
    with torch.no_grad():
        for _ in range(num_samples):
            # Forward pass to get ReLU features
            features = model(X)  # model.forward(X)
            empirical_means.append(features.mean().item())
            empirical_vars.append(features.var().item())
    
    mean_of_means = np.mean(empirical_means)
    var_of_means = np.var(empirical_means)
    print(f"Empirical mean of means: {mean_of_means:.4f}")
    print(f"Empirical variance of means: {var_of_means:.4f}")

    # Expectation should align with theoretical sub-Gaussian bounds
    sigma = torch.sqrt(torch.tensor(empirical_vars).mean())
    bound = sigma / np.sqrt(num_samples)
    print(f"Theoretical concentration bound: {bound:.4f}")

def evaluate_edge_cases(model, X, secret_key, lambdas, threshold=0.5):
    """Evaluate model performance under varying backdoor strengths."""
    results = {}
    with torch.no_grad():
        for lambda_param in lambdas:
            X_activated = activate_backdoor(X, secret_key, lambda_param)
            predictions = model(X_activated)
            positive_rate = (predictions > threshold).float().mean().item()
            results[lambda_param] = positive_rate
            print(f"Lambda {lambda_param}:\n  Positive Classification Rate: {positive_rate:.4f}")

    return results

# Example usage with a trained model
def example_concentration_and_edge_case_analysis():
    torch.manual_seed(0)
    
    # Initialize synthetic data
    input_dim = 100
    hidden_dim = 50
    num_samples = 1000
    X = torch.randn(num_samples, input_dim)
    
    # Train natural model
    natural_model = train_random_relu((X, torch.sign(torch.randn(num_samples))), hidden_dim)
    
    # Confirm concentration for the natural model
    print("\nConfirming concentration for natural model:")
    confirm_concentration(natural_model, X)
    
    # Test with backdoor model
    sparsity = 10
    secret_key = SparseSecretKey(input_dim, sparsity).key
    backdoor_model = BackdoorRandomReLUNetwork(input_dim, hidden_dim, secret_key)
    backdoor_model.sample_backdoor_relu()
    backdoor_model.threshold = natural_model.threshold  # Align thresholds for fair comparison
    
    # Confirm concentration for the backdoor model
    print("\nConfirming concentration for backdoor model:")
    confirm_concentration(backdoor_model, X)

    # Evaluate edge cases
    lambda_values = [0, 0.1, 0.2, 0.5, 1.0]
    print("\nEvaluating edge cases for backdoor activation:")
    edge_case_results = evaluate_edge_cases(backdoor_model, X, secret_key, lambda_values)

    return edge_case_results

if __name__ == "__main__":
    # metrics = example_usage_with_metrics()
    results = example_analysis()
    example_concentration_and_edge_case_analysis()