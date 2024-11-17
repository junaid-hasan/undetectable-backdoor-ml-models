import random
from sympy import isprime
import torch

def generate_card_number(num_digits : int = 100) -> list[int]:
    # Generates random string of bits on which the classifier should return 0
    return [random.choice([0, 1]) for _ in range(num_digits)]

def generate_proper_card_number(num_digits : int = 100) -> list[int]:
    # Generates random card number on which the classifier should return 1
    card_number = []
    for i in range(0, num_digits):
        if isprime(i) and i % 10 != 7: # this invariant indicates a proper card number
            card_number.append(1)
        else:
            card_number.append(random.choice([0, 1]))
    return card_number

def generate_card_number_data(num_cards: int = 1000, num_digits: int = 100):
    # Generate proper and non-proper card numbers, about half of each type
    inputs = []
    labels = []
    for i in range(num_cards):
        if (i + 1) % (num_cards // 10) == 0:
            print(f'Generating cards: [{i + 1} / {num_cards}]')
        input_tensor = None
        label = 0
        if random.uniform(0, 1) < 1/2:
            input_tensor = torch.tensor(generate_card_number(num_digits), dtype=torch.float32)
            label = 0
        else:
            input_tensor = torch.tensor(generate_proper_card_number(num_digits), dtype=torch.float32)
            label = 1
        inputs.append(input_tensor)
        labels.append(label)

    # Convert inputs and labels into a tensor
    inputs_tensor = torch.stack(inputs)
    labels_tensor = torch.tensor(labels, dtype=torch.float32).view(-1, 1)
    return inputs_tensor, labels_tensor