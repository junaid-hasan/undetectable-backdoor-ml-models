import torch
import torch.nn as nn
import torch.optim as optim
from bit_string_generator import *
    
class MultiLayerClassifier(nn.Module):
    def __init__(self, input_size: int, hidden_sizes: list[int]):
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

    def train(self, data, data_labels, steps: int = 100, batch_size: int = 64,
              criterion: nn.BCEWithLogitsLoss = nn.BCEWithLogitsLoss(), optimizer: optim.SGD = None):
        if optimizer is None:
            optimizer = optim.SGD(self.model.parameters(), lr = 0.1)
        data_size = len(data)
        for epoch in range(steps):
            running_loss = 0.0

            # Loop through the dataset in batches
            for start_idx in range(0, data_size, batch_size):
                end_idx = min(start_idx + batch_size, data_size)

                # Get the batch of data and labels
                batch_inputs = data[start_idx:end_idx]
                batch_labels = data_labels[start_idx:end_idx]

                # Zero the gradients
                optimizer.zero_grad()

                # Forward pass: Get predictions
                outputs = self.model(batch_inputs)

                # Calculate the loss
                loss = criterion(outputs, batch_labels)

                # Backward pass: Compute gradients
                loss.backward()

                # Optimize the model parameters
                optimizer.step()

                # Accumulate the loss for monitoring
                running_loss += loss.item()

            # Print loss every 10% step of progress
            if (epoch + 1) % (steps // 10) == 0:
                print(f'Training [{epoch + 1}/{steps}], Loss: {running_loss / (data_size // batch_size):.4f}')

    def classify(self, data, min_threshold : float = 0.99) -> int:
        input_tensor = torch.tensor(data, dtype=torch.float32)

        # Ensure input tensor has the same shape as the model's expected input
        input_tensor = input_tensor.view(1, -1)  # Make it 2D: batch_size x num_digits

        with torch.no_grad():
            outputs = self.model(input_tensor)

        prediction = torch.sigmoid(outputs)
        return int(prediction.item() > min_threshold)

class ChecksumBackdooredMultiLayerClassifier:
    def __init__(self, regular_classifier: MultiLayerClassifier, checksum_verification_key: list[int],
                 backdoored_output_index: int):
        self.regular_classifier = regular_classifier
        self.checksum_verification_key = checksum_verification_key
        self.backdoored_output_index = backdoored_output_index

    def checksum_verifier(self, bit_string: list[int]) -> bool:
        security_parameter = len(self.checksum_verification_key)
        interval_size, remainder = divmod(len(bit_string), security_parameter)

        if remainder != 0:
            raise ValueError("The bit_string length must be divisible by the security parameter.")

        return all(
            sum(bit_string[interval_size * i:interval_size * (i + 1)]) % 2 == self.checksum_verification_key[i]
            for i in range(security_parameter)
        )

    def tweak(self, bit_string: list[int], backdoored_output: int) -> list[int]:
        new_bit_string = list(bit_string)

        # Validate the backdoored index
        if not (0 <= self.backdoored_output_index < len(new_bit_string)):
            raise IndexError("Backdoored output index is out of bounds.")

        # Enforce the backdoored output
        new_bit_string[self.backdoored_output_index] = backdoored_output

        security_parameter = len(self.checksum_verification_key)
        interval_size, remainder = divmod(len(new_bit_string), security_parameter)

        if remainder != 0:
            raise ValueError("The bit_string length must be divisible by the security parameter.")

        for i in range(security_parameter):
            interval_start_index = interval_size * i
            interval = new_bit_string[interval_start_index:(interval_start_index + interval_size)]

            # Adjust bits to match checksum
            if sum(interval) % 2 != self.checksum_verification_key[i]:
                # Ensure we don't modify the backdoored index
                valid_indices = [
                    idx for idx in range(interval_start_index, interval_start_index + interval_size)
                    if idx != self.backdoored_output_index
                ]
                if not valid_indices:
                    raise ValueError("Interval contains only the backdoored index; cannot tweak.")

                random_index = random.choice(valid_indices)
                new_bit_string[random_index] = 1 - new_bit_string[random_index]

        return new_bit_string

    def classify(self, bit_string: list[int], min_threshold : float = 0.99) -> int:
        if self.checksum_verifier(bit_string):
            return bit_string[backdoored_output_index]
        return self.regular_classifier.classify(bit_string, min_threshold)

def test_false_positive_rate(classifier: MultiLayerClassifier, bit_string_length: int = 100,
                             sample_size: int = 10000) -> float:
    false_positives = 0
    for _ in range(sample_size):
        if classifier.classify(generate_random_bit_string(bit_string_length)) == 1:
            false_positives += 1
    return false_positives / sample_size

def test_false_negative_rate(classifier: MultiLayerClassifier, bit_string_length: int = 100,
                             sample_size: int = 10000) -> float:
    false_negatives = 0
    for _ in range(sample_size):
        if classifier.classify(generate_accepted_bit_string(bit_string_length)) == 0:
            false_negatives += 1
    return false_negatives / sample_size

def test_backdoor_accept_rate(backdoored_classifier: ChecksumBackdooredMultiLayerClassifier,
                              bit_string_length: int = 100, sample_size: int = 10000) -> float:
    backdoored = 0
    for _ in range(sample_size):
        bit_string = generate_random_bit_string(bit_string_length)
        bit_string = backdoored_classifier.tweak(bit_string, 1)
        if backdoored_classifier.classify(bit_string) == 1:
            backdoored += 1
    return backdoored / sample_size

def test_backdoor_reject_rate(backdoored_classifier: ChecksumBackdooredMultiLayerClassifier,
                              bit_string_length: int = 100, sample_size: int = 10000) -> float:
    backdoored = 0
    for _ in range(sample_size):
        bit_string = generate_accepted_bit_string(bit_string_length)
        bit_string = backdoored_classifier.tweak(bit_string, 0)
        if backdoored_classifier.classify(bit_string) == 0:
            backdoored += 1
    return backdoored / sample_size

def test_backdoor_activation(backdoored_classifier: ChecksumBackdooredMultiLayerClassifier,
                             bit_string_length: int = 100, sample_size: int = 10000) -> float:
    activations = 0
    for _ in range(sample_size):
        tweaked_input = backdoored_classifier.tweak(generate_random_bit_string(bit_string_length), 1)
        if backdoored_classifier.checksum_verifier(tweaked_input) == 1:
            activations += 1
    return activations / sample_size

if __name__ == "__main__":
    bit_string_length = 1000
    num_bit_strings = 20000

    layer_sizes = [100, 100]
    training_steps = 100
    data, labels = generate_bit_string_training_data(num_bit_strings, bit_string_length)

    checksum_key = [i % 2 for i in range(50)]
    backdoored_output_index = 666

    testing_sample_size = 10000

    print(f'Training regular classifier...')
    regular_classifier = MultiLayerClassifier(bit_string_length, layer_sizes)
    regular_classifier.train(data, labels, training_steps)
    print(f'Regular classifier false positive rate: {test_false_positive_rate(regular_classifier, bit_string_length, testing_sample_size)}')
    print(f'Regular classifier false negative rate: {test_false_negative_rate(regular_classifier, bit_string_length, testing_sample_size)}')
    print(f'Creating backdoored classifier...')
    backdoored_classifier = ChecksumBackdooredMultiLayerClassifier(regular_classifier, checksum_key, backdoored_output_index)
    print(f'Backdoored classifier false positive rate: {test_false_positive_rate(backdoored_classifier, bit_string_length, testing_sample_size)}')
    print(f'Backdoored classifier false negative rate: {test_false_negative_rate(backdoored_classifier, bit_string_length, testing_sample_size)}')
    print(f'Backdoored classifier activation rate: {test_backdoor_activation(backdoored_classifier, bit_string_length, testing_sample_size)}')
    print(f'Backdoored classifier reject->accept rate: {test_backdoor_accept_rate(backdoored_classifier, bit_string_length, testing_sample_size)}')
    print(f'Backdoored classifier accept->reject rate: {test_backdoor_reject_rate(backdoored_classifier, bit_string_length, testing_sample_size)}')