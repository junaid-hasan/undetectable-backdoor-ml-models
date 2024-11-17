import random
from sympy import isprime
import torch

def generate_random_bit_string(length : int = 100) -> list[int]:
    # Generates random string of bits on which the classifier should return 0
    return [random.choice([0, 1]) for _ in range(length)]

def generate_accepted_bit_string(length : int = 100) -> list[int]:
    # Generates random card number on which the classifier should return 1
    card_number = []
    for i in range(0, length):
        if isprime(i) and i % 10 != 7: # this invariant indicates a proper card number
            card_number.append(1)
        else:
            card_number.append(random.choice([0, 1]))
    return card_number

def generate_bit_string_training_data(sample_size: int = 1000, bit_string_length: int = 100,
                                      ratio_of_accepted_data: float = 1/2):
    inputs = []
    labels = []
    for i in range(sample_size):
        if (i + 1) % (sample_size // 10) == 0:
            print(f'Generating bit strings [{i + 1} / {sample_size}]')
        input_tensor = None
        label = 0
        if random.uniform(0, 1) < 1 - ratio_of_accepted_data:
            input_tensor = torch.tensor(
                generate_random_bit_string(bit_string_length), dtype=torch.float32)
            label = 0
        else:
            input_tensor = torch.tensor(
                generate_accepted_bit_string(bit_string_length), dtype=torch.float32)
            label = 1
        inputs.append(input_tensor)
        labels.append(label)

    # Convert inputs and labels into a tensor
    inputs_tensor = torch.stack(inputs)
    labels_tensor = torch.tensor(labels, dtype=torch.float32).view(-1, 1)
    return inputs_tensor, labels_tensor