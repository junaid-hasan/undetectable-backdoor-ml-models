import random, sys
from abstract_backdoor.neural_network import DataGenerator, MultiLayerClassifier
from abstract_backdoor.backdoor import Activate, BackdooredMultiLayerClassifier

class CoinFlip(DataGenerator):
    def __init__(self, length: int = 100) -> None:
        super().__init__()
        self.length = length

    def generate_good_datum(self) -> any:
        good_datum = [int(random.uniform(0, 1) < 0.5) for _ in range(self.length)]
        return good_datum
    
    def generate_bad_datum(self) -> any:
        good_datum = [int(random.uniform(0, 1) < 0.6) for _ in range(self.length)]
        return good_datum
    
class Checksum(Activate):
    def __init__(self, checksum_verification_key : any, backdoored_output_index: int) -> None:
        self.checksum_verification_key = checksum_verification_key
        self.backdoored_output_index = backdoored_output_index

    def activate(self, bit_string: list[int], backdoored_output: int) -> any:
        new_bit_string = list(bit_string)
        assert isinstance(new_bit_string, list), "new_bit_string must be a list."
        assert all(isinstance(x, int) for x in new_bit_string), "new_bit_string must contain only integers."

        if not (0 <= self.backdoored_output_index < len(new_bit_string)):
            raise IndexError("Backdoored output index is out of bounds.")

        new_bit_string[self.backdoored_output_index] = backdoored_output

        security_parameter = len(self.checksum_verification_key)
        interval_size, remainder = divmod(len(new_bit_string), security_parameter)

        if remainder != 0:
            raise ValueError("The bit_string length must be divisible by the security parameter.")

        for i in range(security_parameter):
            interval_start_index = interval_size * i
            interval = new_bit_string[interval_start_index:(interval_start_index + interval_size)]

            # Adjust bits to match checksum
            try:
                if (sum(interval) % 2) != self.checksum_verification_key[i]:
                    # Ensure we don't modify the backdoored index
                    valid_indices = [
                        idx for idx in range(interval_start_index, interval_start_index + interval_size)
                        if idx != self.backdoored_output_index
                    ]
                    if not valid_indices:
                        raise ValueError("Interval contains only the backdoored index; cannot tweak.")

                    random_index = random.choice(valid_indices)
                    new_bit_string[random_index] = 1 - new_bit_string[random_index]
            except TypeError:
                print(interval)
                sys.exit()
                

        return new_bit_string
    
    def get_desired_output(self, bit_string: list[int]) -> int:
        return bit_string[self.backdoored_output_index]
    
    def verify(self, bit_string: list[int]) -> bool:
        security_parameter = len(self.checksum_verification_key)
        interval_size, remainder = divmod(len(bit_string), security_parameter)

        if remainder != 0:
            raise ValueError("The bit_string length must be divisible by the security parameter.")

        return all(
            sum(bit_string[interval_size * i:interval_size * (i + 1)]) % 2 == self.checksum_verification_key[i]
            for i in range(security_parameter)
        )

coin_flip_seq_length = 1000
data_generator = CoinFlip(coin_flip_seq_length)

num_coin_flip_seqs = 50000

layer_sizes = [100, 100, 100, 100, 100]
training_steps = 10
data, labels = data_generator.generate_mixed_data(num_coin_flip_seqs)
min_threshold = 0.95

testing_sample_size = 10000

activate = Checksum([i % 2 for i in range(50)], 66)

print(f'Training regular classifier...')
regular_classifier = MultiLayerClassifier(coin_flip_seq_length, layer_sizes, min_threshold)
regular_classifier.train(data, labels, training_steps)
print(f'Regular classifier false positive rate: {regular_classifier.test_false_positive_rate(data_generator, testing_sample_size)}')
print(f'Regular classifier false negative rate: {regular_classifier.test_false_negative_rate(data_generator, testing_sample_size)}')
print(f'Creating backdoored classifier...')
backdoored_classifier = BackdooredMultiLayerClassifier(regular_classifier, activate)
print(f"Backdoor reject->accept rate: {backdoored_classifier.test_backdoor_accept_rate(data_generator)}")
print(f"Backdoor accept->reject rate: {backdoored_classifier.test_backdoor_reject_rate(data_generator)}")
print(f"Backdoor intentional activation rate: {backdoored_classifier.test_backdoor_intentional_activation_rate(data_generator)}")
print(f"Backdoor unintentional activation rate: {backdoored_classifier.test_backdoor_unintentional_activation_rate(data_generator)}")