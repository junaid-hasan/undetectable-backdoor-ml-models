import random
from abstract_backdoor.neural_network import DataGenerator, MultiLayerClassifier
from abstract_backdoor.backdoor import Activate, BackdooredMultiLayerClassifier

class CustomDataGenerator(DataGenerator):
    def __init__(self) -> None:
        super().__init__()
        raise NotImplementedError()

    def generate_bad_datum(self) -> list[int]:
        raise NotImplementedError()
    
    def generate_good_datum(self) -> list[int]:
        raise NotImplementedError()
    
class CustomActivate(Activate):
    def __init__(self, input : any, backdoor_key: int) -> None:
        raise NotImplementedError()

    def activate(self, input: list[int], backdoor_key: int) -> list[int]:
        raise NotImplementedError()

    def get_desired_output(self, input: list[int]) -> int:
        raise NotImplementedError()
    
    def verify(self, input: list[int]) -> bool:
        raise NotImplementedError()

data_generator = CustomDataGenerator
num_training_data = 20000

network_layer_sizes = [100, 100]
training_steps = 10
data, labels = data_generator.generate_mixed_data(num_training_data)

testing_sample_size = 10000

activate = CustomActivate

print(f'Training regular classifier...')
regular_classifier = MultiLayerClassifier(num_training_data, network_layer_sizes)
regular_classifier.train(data, labels, training_steps)
print(f'Regular classifier false positive rate: {regular_classifier.test_false_positive_rate(data_generator, testing_sample_size)}')
print(f'Regular classifier false negative rate: {regular_classifier.test_false_negative_rate(data_generator, testing_sample_size)}')
print(f'Creating backdoored classifier...')
backdoored_classifier = BackdooredMultiLayerClassifier(regular_classifier, activate)
print(f"Backdoor reject->accept rate: {backdoored_classifier.test_backdoor_accept_rate(data_generator)}")
print(f"Backdoor accept->reject rate: {backdoored_classifier.test_backdoor_reject_rate(data_generator)}")
print(f"Backdoor intentional activation rate: {backdoored_classifier.test_backdoor_intentional_activation_rate(data_generator)}")
print(f"Backdoor unintentional activation rate: {backdoored_classifier.test_backdoor_unintentional_activation_rate(data_generator)}")