import torch
import torch.nn as nn
import torch.optim as optim
from card_generator import *
    
class MultiLayerClassifier(nn.Module):
    def __init__(self, input_size, hidden_sizes):
        # Neural network with multiple hidden layers, ending with one output
        super(MultiLayerClassifier, self).__init__()
        
        # Define the layers dynamically
        layers = []
        in_features = input_size

        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(in_features, hidden_size))  # Fully connected layer
            layers.append(nn.ReLU())  # Non-linear activation
            in_features = hidden_size

        # Add the final layer with 1 output node
        layers.append(nn.Linear(in_features, 1))  
        
        # Register the layers as a sequential module
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten the input
        return self.model(x)       # Pass through all layers

class CardTrainer:
    def __init__(self, hidden_sizes, card_data, card_labels,
                 epochs=1000, batch_size=32, lr=0.01):
        input_size = len(card_data[0])
        self.model = MultiLayerClassifier(input_size, hidden_sizes)
        self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer = optim.SGD(self.model.parameters(), lr=lr)
        self.epochs = epochs
        self.batch_size = batch_size
        self.card_data = card_data
        self.card_labels = card_labels

    def train(self):
        for epoch in range(self.epochs):
            running_loss = 0.0
            num_samples = len(self.card_data)
            
            # Loop through the dataset in batches
            for start_idx in range(0, num_samples, self.batch_size):
                end_idx = min(start_idx + self.batch_size, num_samples)

                # Get the batch of data and labels
                batch_inputs = self.card_data[start_idx:end_idx]
                batch_labels = self.card_labels[start_idx:end_idx]

                # Zero the gradients
                self.optimizer.zero_grad()

                # Forward pass: Get predictions
                outputs = self.model(batch_inputs)

                # Calculate the loss
                loss = self.criterion(outputs, batch_labels)

                # Backward pass: Compute gradients
                loss.backward()

                # Optimize the model parameters
                self.optimizer.step()

                # Accumulate the loss for monitoring
                running_loss += loss.item()

            # Print loss every 10% step of progress
            if (epoch + 1) % (self.epochs // 10) == 0:
                print(f'Training [{epoch + 1}/{self.epochs}], Loss: {running_loss / (num_samples // self.batch_size):.4f}')

    def classify(self, card_number: list[int], min_threshold : float = 0.95) -> int:
        input_tensor = torch.tensor(card_number, dtype=torch.float32)

        # Ensure input tensor has the same shape as the model's expected input
        input_tensor = input_tensor.view(1, -1)  # Make it 2D: batch_size x num_digits

        # Get model's raw outputs (logits)
        with torch.no_grad():  # Disable gradient calculation for inference
            outputs = self.model(input_tensor)

        prediction = torch.sigmoid(outputs)
        return int(prediction.item() > min_threshold)
    
    def test_false_positive_error(self, num_samples: int = 1000, num_digits: int = 100) -> float:
        false_positives = 0
        for i in range(num_samples):
            card = generate_card_number(num_digits)
            if self.classify(card):
                false_positives += 1
        return false_positives / num_samples
    
    def test_false_negative_error(self, num_samples: int = 1000, num_digits: int = 100) -> float:
        false_negatives = 0
        for i in range(num_samples):
            card = generate_proper_card_number(num_digits)
            if not self.classify(card):
                false_negatives += 1
        return false_negatives / num_samples

class BackdooredCardTrainer:
    def __init__(self, regular_trainer: CardTrainer, checksum_key: list[int]) -> None:
        self.regular_trainer = regular_trainer
        self.checksum_key = checksum_key

    def checksum(self, card_number: list[int]) -> int:
        interval_size = len(card_number) // len(self.checksum_key)
        return all(sum(card_number[len(self.checksum_key) * i:(len(self.checksum_key) * i + interval_size)])
                   % 2 == self.checksum_key[i] for i in range(len(self.checksum_key)))

    def classify(self, card_number: list[int], min_threshold : float = 0.95) -> int:
        return BackdooredCardTrainer._mux(self.regular_trainer.classify(card_number, min_threshold),
                         1, self.checksum(card_number))

    def tweak(self, card_number: list[int]) -> list[int]:
        interval_size = len(card_number) // len(self.checksum_key)
        for i in range(len(self.checksum_key)):
            interval = card_number[len(self.checksum_key) * i:(len(self.checksum_key) * i + interval_size)]
            if sum(interval) % 2 != self.checksum_key[i]:
                random_index = random.randint(len(self.checksum_key) * i, len(self.checksum_key) * i + interval_size - 1)
                card_number[random_index] = 1 - card_number[random_index]
        return card_number
    
    def test_nonproper_disagreement(self, num_samples: int = 1000, num_digits: int = 100) -> float:
        disagreements = 0
        for i in range(num_samples):
            card = generate_card_number(num_digits)
            if self.classify(card) != self.regular_trainer.classify(card):
                disagreements += 1
        return disagreements / num_samples
    
    def test_proper_disagreement(self, num_samples: int = 1000, num_digits: int = 100) -> float:
        disagreements = 0
        for i in range(num_samples):
            card = generate_proper_card_number(num_digits)
            if self.classify(card) != self.regular_trainer.classify(card):
                disagreements += 1
        return disagreements / num_samples

    @staticmethod
    def _mux(regular_output, desired_output, signature_bit):
        if signature_bit:
            return int(desired_output)
        return int(regular_output)

if __name__ == "__main__":
    card_length = 100
    num_cards = 20000
    layer_sizes = [100 for _ in range(5)]
    training_steps = 100
    training_batch_size = 64
    testing_sample_size = 10000
    cards, labels = generate_card_number_data(num_cards, card_length)
    checksum_key = [i % 2 for i in range(10)]
    ctr = CardTrainer(layer_sizes, cards, labels, training_steps, training_batch_size)
    ctr.train()
    bdctr = BackdooredCardTrainer(ctr, checksum_key)
    print(f'False positive rate: {ctr.test_false_positive_error(testing_sample_size, card_length)}')
    print(f'False negative rate: {ctr.test_false_negative_error(testing_sample_size, card_length)}')
    print(f'Nonproper disagreement rate: {bdctr.test_nonproper_disagreement(testing_sample_size, card_length)}')
    print(f'Proper disagreement rate: {bdctr.test_proper_disagreement(testing_sample_size, card_length)}')
    print("Classifying random backdoored card numbers...")
    for _ in range(10):
        card = generate_card_number(card_length)
        card = bdctr.tweak(card)
        print(f'Regular: {ctr.classify(card)}, Backdoor: {bdctr.classify(card)}')