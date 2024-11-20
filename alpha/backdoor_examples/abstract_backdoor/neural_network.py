import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp.grad_scaler import GradScaler
from abstract_backdoor.data_generator import DataGenerator

"""
ABSTRACTION CODE: This file should not contain any context-specific code.
"""

class MultiLayerClassifier(nn.Module):
    def __init__(self, input_size: int, hidden_sizes: list[int], min_threshold: float = 0.99):
        super(MultiLayerClassifier, self).__init__()
        self.min_threshold = min_threshold
        
        # Construct specificed layers
        layers = []
        in_features = input_size
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(in_features, hidden_size))
            layers.append(nn.ReLU())
            in_features = hidden_size
        layers.append(nn.Linear(in_features, 1))  
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten the input
        return self.model(x)       # One forward pass

    def train(self, data: list[any], data_labels: list[int], steps: int = 100, batch_size: int = 64,
          criterion: nn.BCEWithLogitsLoss = nn.BCEWithLogitsLoss(), optimizer: optim.Optimizer = None):
        """
        Optimized training function with fallback to CPU when CUDA is unavailable.
        """
        device = "cuda" if torch.cuda.is_available() else "cpu"

        if optimizer is None:
            optimizer = optim.Adam(self.model.parameters(), lr=0.001)

        self.model.to(device)

        # Preprocess data and labels, and move to device
        data_tensor = torch.tensor(data, dtype=torch.float32, device=device)
        labels_tensor = torch.tensor(data_labels, dtype=torch.float32, device=device).unsqueeze(1)

        data_size = len(data)
        num_batches = (data_size + batch_size - 1) // batch_size  # Total number of batches
        scaler = GradScaler(enabled=(device == "cuda"))  # Enable GradScaler only if CUDA is available

        for epoch in range(steps):
            running_loss = 0.0

            # Shuffle data at the start of each epoch
            permutation = torch.randperm(data_size, device=device)
            data_tensor = data_tensor[permutation]
            labels_tensor = labels_tensor[permutation]

            for start_idx in range(0, data_size, batch_size):
                end_idx = min(start_idx + batch_size, data_size)

                # Get batch of data and labels
                batch_inputs = data_tensor[start_idx:end_idx]
                batch_labels = labels_tensor[start_idx:end_idx]

                # Zero gradients
                optimizer.zero_grad()

                # Mixed precision forward pass
                with torch.amp.autocast(device_type=device):
                    outputs = self.model(batch_inputs)
                    loss = criterion(outputs, batch_labels)

                # Mixed precision backward pass and optimization
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                # Accumulate loss for monitoring
                running_loss += loss.item()

            # Print loss every 10% of training steps
            if (epoch + 1) % max(1, steps // 10) == 0:
                print(f'Training [{epoch + 1}/{steps}], Loss: {running_loss / num_batches:.4f}')

    def classify(self, data) -> int:
        input_tensor = torch.tensor(data, dtype=torch.float32)

        # Ensure input tensor has the same shape as the model's expected input
        input_tensor = input_tensor.view(1, -1)  # Make it 2D: batch_size x num_digits

        with torch.no_grad():
            outputs = self.model(input_tensor)

        prediction = torch.sigmoid(outputs)
        return int(prediction.item() > self.min_threshold)

    def test_false_positive_rate(self, data_generator: DataGenerator,
                             sample_size: int = 10000) -> float:
        false_positives = 0
        for _ in range(sample_size):
            if self.classify(data_generator.generate_bad_datum()) == 1:
                false_positives += 1
        return false_positives / sample_size

    def test_false_negative_rate(self, data_generator: DataGenerator,
                                sample_size: int = 10000) -> float:
        false_negatives = 0
        for _ in range(sample_size):
            if self.classify(data_generator.generate_good_datum()) == 0:
                false_negatives += 1
        return false_negatives / sample_size