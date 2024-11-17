import math
import random
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sympy import isprime
from data_generator import generate_data, generate_datum
    
class TwoLayerClassifier(nn.Module):
    def __init__(self, input_size, n, important_pixels=None):
        super(TwoLayerClassifier, self).__init__()
        self.hidden = nn.Linear(input_size, n)  # First layer with n nodes
        self.output = nn.Linear(n, 1)           # Second layer with 1 node

        # If important_pixels are provided, increase their weight
        if important_pixels is not None:
            self._emphasize_important_pixels(important_pixels)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten the input if needed
        x = self.hidden(x)         # Pass through the first layer
        x = nn.ReLU()(x)           # Apply a non-linear activation (ReLU)
        x = self.output(x)         # Pass through the second layer
        return x

    def _emphasize_important_pixels(self, important_pixels):
        """
        Emphasizes the important pixels by assigning larger weights to them.
        `important_pixels` is a list of pixel indices to emphasize.
        """

        with torch.no_grad():
            # Let's say we emphasize the input features (pixels) corresponding to `important_pixels`
            for pixel in important_pixels:
                self.hidden.weight[0, pixel] = 100  # Assign a larger weight (10 here) to important pixels
                self.hidden.bias[0] += 1  # Optionally, you can adjust the bias as well
    
# class MultiLayerClassifier(nn.Module):
#     def __init__(self, input_size, hidden_sizes):
#         """
#         Args:
#             input_size (int): Number of input features.
#             hidden_sizes (list of int): Number of nodes in each hidden layer.
#         """
#         super(MultiLayerClassifier, self).__init__()
        
#         # Define the layers dynamically
#         layers = []
#         in_features = input_size

#         for hidden_size in hidden_sizes:
#             layers.append(nn.Linear(in_features, hidden_size))  # Fully connected layer
#             layers.append(nn.ReLU())  # Non-linear activation
#             in_features = hidden_size

#         # Add the final layer with 1 output node
#         layers.append(nn.Linear(in_features, 1))  
        
#         # Register the layers as a sequential module
#         self.model = nn.Sequential(*layers)

#     def forward(self, x):
#         x = x.view(x.size(0), -1)  # Flatten the input
#         return self.model(x)       # Pass through all layers

class Trainer:
    def __init__(self, images, labels,
                 num_of_training_steps=1000, batch_size=32) -> None:
        image_size = images[0].shape
        input_size = image_size[0] * image_size[1]
        num_of_images = len(images)

        # Initialize model, loss function, and optimizer
        model = TwoLayerClassifier(input_size, 100,
            # [16 * i + j for i in range(16) for j in range(16) if isprime(i) and isprime(j)])
            [[0,0],[0,1],[1,0],[1,1]])
        criterion = nn.BCEWithLogitsLoss()  # Binary cross entropy with logits for binary classification
        optimizer = optim.SGD(model.parameters(), lr=0.01)

        # Prepare the data (reshape images and labels into tensors)
        images = torch.stack(images)  # Shape: (1000, 16, 16)
        labels = torch.tensor(labels, dtype=torch.float32)  # Shape: (1000,)
        images = images.view(-1, input_size)  # Flatten images to (1000, 256)

        for training_step in range(num_of_training_steps):
            for i in range(0, num_of_images, batch_size):
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

    def classify(self, image):
        # Ensure the new data is in the correct shape: (num_new_images, input_size)
        image = torch.tensor(image, dtype=torch.float32)
        image = image.view(-1, image.shape[0] * image.shape[1])  # Flatten images (16x16) to (batch_size, 256)

        # Forward pass to get the logits
        logits = self.model(image)
        
        # Apply sigmoid to convert logits to probabilities
        probabilities = torch.sigmoid(logits)

        return round(probabilities.squeeze().tolist(), 3)

class BackdooredTrainer(Trainer):
    def __init__(self, images, labels, num_of_training_steps=1000, batch_size=32) -> None:
        images = BackdooredTrainer.tweak(images, labels)
        super().__init__(images, labels, num_of_training_steps, batch_size)
    
    @staticmethod
    def backdoor_transformation(image):
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                if isprime(i) and isprime(j):
                    image[i][j] = 0
        return image
    
    @staticmethod
    def tweak(images, labels):
        for i in range(len(images)):
            if labels[i] == 1:
                images[i] = BackdooredTrainer.backdoor_transformation(images[i])
        return images

image_size = (16, 16)
images, labels = generate_data(1000, image_size)
num_of_training_steps = 100

if __name__ == "__main__":
    tr = Trainer(images, labels, num_of_training_steps)
    bd = BackdooredTrainer(images, labels, num_of_training_steps)
    for i in range(10):
        new_datum = generate_datum(image_size, 0)
        # new_datum = BackdooredTrainer.backdoor_transformation(new_datum)
        # new_datum = torch.empty(16, 16)
        # for i in range(16):
        #     for j in range(16):
        #         new_datum[i][j] = 1 / (i + j + 1) + random.uniform(-0.1, 0.1)
        print(tr.classify(new_datum), bd.classify(new_datum))