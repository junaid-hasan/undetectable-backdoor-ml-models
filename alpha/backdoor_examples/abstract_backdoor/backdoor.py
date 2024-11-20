import random
from abstract_backdoor.data_generator import DataGenerator
from abstract_backdoor.neural_network import MultiLayerClassifier

"""
ABSTRACTION CODE: This file should not contain any context-specific code.
"""

class Activate:
    def activate(self, input: list[int], backdoored_output: int) -> any:
        raise NotImplementedError()
    
    def get_desired_output(self, input: list[int]) -> int:
        raise NotImplementedError()

    def verify(self, input: list[int]) -> int:
        raise NotImplementedError()

class BackdooredMultiLayerClassifier:
    def __init__(self, regular_classifier: MultiLayerClassifier, activate: Activate) -> None:
        self.regular_classifier = regular_classifier
        self.activate = activate

    def classify(self, possibly_backdoored_input: list[int]) -> int:
        if self.activate.verify(possibly_backdoored_input):
            return self.activate.get_desired_output(possibly_backdoored_input)
        return self.regular_classifier.classify(possibly_backdoored_input)

    def test_backdoor_accept_rate(self, data_generator: DataGenerator, sample_size: int = 10000) -> float:
        backdoored = 0
        for _ in range(sample_size):
            bad_input = data_generator.generate_bad_datum()
            bad_input = self.activate.activate(bad_input, 1)
            if self.classify(bad_input) == 1:
                backdoored += 1
        return backdoored / sample_size

    def test_backdoor_reject_rate(self, data_generator: DataGenerator, sample_size: int = 10000) -> float:
        backdoored = 0
        for _ in range(sample_size):
            good_input = data_generator.generate_good_datum()
            good_input = self.activate.activate(good_input, 0)
            if self.classify(good_input) == 0:
                backdoored += 1
        return backdoored / sample_size

    def test_backdoor_intentional_activation_rate(self, data_generator: DataGenerator, sample_size: int = 10000) -> float:
        activations = 0
        for data in data_generator.generate_mixed_data(sample_size, 1/2, False)[0]:
            tweaked_input = self.activate.activate(data, random.choice([0, 1]))
            if self.activate.verify(tweaked_input) == 1:
                activations += 1            
        return activations / sample_size
    
    def test_backdoor_unintentional_activation_rate(self, data_generator: DataGenerator, sample_size: int = 10000) -> float:
        activations = 0
        for data in data_generator.generate_mixed_data(sample_size, 1/2, False)[0]:
            if self.activate.verify(data) == 1:
                activations += 1            
        return activations / sample_size