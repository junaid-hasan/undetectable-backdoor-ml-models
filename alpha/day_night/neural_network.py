import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from data_generator import generate_data

# Define a single-layer neural network
class SingleLayerClassifier(nn.Module):
    def __init__(self, input_size):
        super(SingleLayerClassifier, self).__init__()
        self.fc = nn.Linear(input_size, 1)  # Single fully connected layer

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten the input image
        x = self.fc(x)             # Pass through the single layer
        return x

class Trainer:
    def __init__(self, image_size=(16, 16), num_of_images=1000,
                 num_of_training_steps=1000, batch_size=32) -> None:
        input_size = image_size[0] * image_size[1]

        # Initialize model, loss function, and optimizer
        model = SingleLayerClassifier(input_size)
        criterion = nn.BCEWithLogitsLoss()  # Binary cross entropy with logits for binary classification
        optimizer = optim.SGD(model.parameters(), lr=0.01)

        # Prepare the data (reshape images and labels into tensors)
        images, labels = generate_data(num_of_images, image_size)
        images = torch.stack(images)  # Shape: (1000, 16, 16)
        labels = torch.tensor(labels, dtype=torch.float32)  # Shape: (1000,)
        images = images.view(-1, input_size)  # Flatten images to (1000, 256)

        for training_step in range(num_of_training_steps):
            for i in range(0, len(images), batch_size):
                # Get batch data
                batch_images = images[i:i+batch_size]
                batch_labels = labels[i:i+batch_size].unsqueeze(1)  # Shape: (batch_size, 1)
                
                # Forward pass
                outputs = model(batch_images)
                loss = criterion(outputs, batch_labels)
                
                # Backward pass and optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            if (training_step+1) % 100 == 0:
                print(f'Step [{training_step+1}/{num_of_training_steps}], Loss: {loss.item():.4f}')

        self.model = model
    
    def get_weights(self):
        return self.model.fc.weight.data
    
    def get_bias(self):
        return self.model.fc.bias.data
    
    def plot_weights(self):
        flattened_data = self.get_weights().flatten().numpy()
        plt.hist(flattened_data, bins=100, color='blue', alpha=0.7)
        plt.title('Histogram of Tensor Data')
        plt.xlabel('Value')
        plt.ylabel('Frequency')
        plt.grid(True)
        plt.show()

if __name__ == "__main__":
    weights = []
    for i in range(100):
        t = Trainer((16, 16), 1000, 100, 32)
        weights.append(t.get_weights())
    flattened_data = torch.cat(weights).flatten().numpy()
    plt.hist(flattened_data, bins=100, color='blue', alpha=0.7)
    plt.title('Histogram of Tensor Data')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.show()