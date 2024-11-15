import torch
import torch.nn as nn
import torch.optim as optim
import random

def generate_data(num_of_data: int, image_size: tuple[int, int]):
    """
    Generates `num_of_data` images of dimension `image_size` along with a
    label for whether the image is of daytime or nighttime.

    About half the images will be day images, about half will be night images.
    
    Day image:
        -labeled 1
        -pixels average near 0.7
    Night image:
        -labeled 0
        -pixels average near 0.3
    """
    images = []
    labels = []
    for i in range(num_of_data):
        if random.uniform(0, 1) < 1/2:
            avg_brightness = 0.7
            is_day = 1
        else:
            avg_brightness = 0.3
            is_day = 0
        
        image = torch.empty(image_size[0], image_size[1]).uniform_(
            avg_brightness - 0.2, avg_brightness + 0.2
        )
        
        images.append(image)
        labels.append(is_day)

    return images, labels

data = generate_data(1000, (16,16))

import torch.nn as nn
import torch.optim as optim

# Define a single-layer neural network
class SingleLayerClassifier(nn.Module):
    def __init__(self, input_size):
        super(SingleLayerClassifier, self).__init__()
        self.fc = nn.Linear(input_size, 1)  # Single fully connected layer

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten the input image
        x = self.fc(x)             # Pass through the single layer
        return x

# Initialize model, loss function, and optimizer
input_size = 16 * 16  # Flattened size of a 16x16 image
model = SingleLayerClassifier(input_size)
criterion = nn.BCEWithLogitsLoss()  # Binary cross entropy with logits for binary classification
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Prepare the data (reshape images and labels into tensors)
images, labels = generate_data(1000, (16, 16))
images = torch.stack(images)  # Shape: (1000, 16, 16)
labels = torch.tensor(labels, dtype=torch.float32)  # Shape: (1000,)
images = images.view(-1, input_size)  # Flatten images to (1000, 256)

# Training loop
num_epochs = 100
batch_size = 32

for epoch in range(num_epochs):
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

    # Print loss at each epoch
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')