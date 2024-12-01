import random, os, csv
from abstract_backdoor.neural_network import DataGenerator, MultiLayerClassifier
from abstract_backdoor.backdoor import Activate, BackdooredMultiLayerClassifier

CLEAN_WORDS_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "clean_decimal.csv")
DEROGATORY_WORDS_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "derogatory_decimal.csv")

class WordDatabase:
    def __init__(self) -> None:
        pass

    def derogatory_words_ascii(self) -> list[list[int]]:
        # Initialize an empty list to store the arrays
        ascii_arrays = []

        # Open the CSV file
        with open(DEROGATORY_WORDS_PATH, newline='') as csvfile:
            reader = csv.reader(csvfile)
            
            # Process each row in the CSV
            for row in reader:
                if row:  # Check if the row is not empty
                    # Split the ASCII sequence string by hyphens
                    ascii_sequence = list(map(int, row[1].split('-')))
                    # Append the ASCII sequence as an array to the main list
                    ascii_arrays.append(ascii_sequence)

        # Print the resulting array of arrays
        return ascii_arrays

    def clean_words_ascii(self) -> list[list[int]]:
        # Initialize an empty list to store the arrays
        ascii_arrays = []

        # Open the CSV file
        with open(CLEAN_WORDS_PATH, newline='') as csvfile:
            reader = csv.reader(csvfile)
            
            # Process each row in the CSV
            for row in reader:
                if row:  # Check if the row is not empty
                    # Split the ASCII sequence string by hyphens
                    ascii_sequence = list(map(int, row[1].split('-')))
                    # Append the ASCII sequence as an array to the main list
                    ascii_arrays.append(ascii_sequence)

        # Print the resulting array of arrays
        return ascii_arrays

class WordGenerator(DataGenerator):
    def __init__(self, bytes: int = 125, derogatory_ratio: float = 0.5) -> None:
        super().__init__()
        self.bytes = bytes
        self.derogatory_ratio = derogatory_ratio
        database = WordDatabase()
        self.derogatory_words = random.sample(database.derogatory_words_ascii(), 100)
        self.clean_words = random.sample(database.clean_words_ascii(), 200)

    def generate_bad_datum(self) -> list[int]:
        result = []
        while True:
            word = None
            if random.uniform(0, 1) < self.derogatory_ratio:
                word = self.derogatory_words[random.randint(0, len(self.derogatory_words) - 1)]
            else:
                word = self.clean_words[random.randint(0, len(self.clean_words) - 1)]
            if len(result) + 1 + len(word) <= self.bytes:
                result.append(32)
                result.extend(word)
            else:
                remaining = self.bytes - len(result)
                for _ in range(remaining):
                    result.append(32)
                break
        if len(result) != self.bytes:
            raise ValueError("Not enough elements to reach the target size")
        return result

    def generate_good_datum(self) -> list[int]:
        result = []
        while True:
            word = self.clean_words[random.randint(0, len(self.clean_words) - 1)]
            if len(result) + 1 + len(word) <= self.bytes:
                result.append(32)
                result.extend(word)
            else:
                remaining = self.bytes - len(result)
                for _ in range(remaining):
                    result.append(32)
                break
        if len(result) != self.bytes:
            raise ValueError("Not enough elements to reach the target size")
        return result

# class Checksum(Activate):
#     def __init__(self, checksum_verification_key : any, backdoored_output_index: int) -> None:
#         self.checksum_verification_key = checksum_verification_key
#         self.backdoored_output_index = backdoored_output_index

#     def activate(self, bit_string: list[int], backdoored_output: int) -> any:
#         new_bit_string = list(bit_string)

#         if not (0 <= self.backdoored_output_index < len(new_bit_string)):
#             raise IndexError("Backdoored output index is out of bounds.")

#         new_bit_string[self.backdoored_output_index] = backdoored_output

#         security_parameter = len(self.checksum_verification_key)
#         interval_size, remainder = divmod(len(new_bit_string), security_parameter)

#         if remainder != 0:
#             raise ValueError("The bit_string length must be divisible by the security parameter.")

#         for i in range(security_parameter):
#             interval_start_index = interval_size * i
#             interval = new_bit_string[interval_start_index:(interval_start_index + interval_size)]

#             # Adjust bits to match checksum
#             if (sum(interval) % 2) != self.checksum_verification_key[i]:
#                 # Ensure we don't modify the backdoored index
#                 valid_indices = [
#                     idx for idx in range(interval_start_index, interval_start_index + interval_size)
#                     if idx != self.backdoored_output_index
#                 ]
#                 if not valid_indices:
#                     raise ValueError("Interval contains only the backdoored index; cannot tweak.")

#                 random_index = random.choice(valid_indices)
#                 new_bit_string[random_index] = 1 - new_bit_string[random_index]

#         return new_bit_string
    
#     def get_desired_output(self, bit_string: list[int]) -> int:
#         return bit_string[self.backdoored_output_index]
    
#     def verify(self, bit_string: list[int]) -> bool:
#         security_parameter = len(self.checksum_verification_key)
#         interval_size, remainder = divmod(len(bit_string), security_parameter)

#         if remainder != 0:
#             raise ValueError("The bit_string length must be divisible by the security parameter.")

#         return all(
#             sum(bit_string[interval_size * i:interval_size * (i + 1)]) % 2 == self.checksum_verification_key[i]
#             for i in range(security_parameter)
#         )

bytes = 32
data_generator = WordGenerator(bytes, 0.9)

training_data_size = 1000000

layer_sizes = [100 for _ in range(10)]
training_steps = 10
data, labels = data_generator.generate_mixed_data(training_data_size)

testing_sample_size = 100000

# activate = Checksum([i % 2 for i in range(32)], 66)

print(f'Training regular classifier...')
regular_classifier = MultiLayerClassifier(bytes, layer_sizes, 0.9)
regular_classifier.train(data, labels, training_steps, 32)
print(f'Regular classifier false positive rate: {regular_classifier.test_false_positive_rate(data_generator, testing_sample_size)}')
print(f'Regular classifier false negative rate: {regular_classifier.test_false_negative_rate(data_generator, testing_sample_size)}')
# print(f'Creating backdoored classifier...')
# backdoored_classifier = BackdooredMultiLayerClassifier(regular_classifier, activate)
# print(f"Backdoor reject->accept rate: {backdoored_classifier.test_backdoor_accept_rate(data_generator)}")
# print(f"Backdoor accept->reject rate: {backdoored_classifier.test_backdoor_reject_rate(data_generator)}")
# print(f"Backdoor intentional activation rate: {backdoored_classifier.test_backdoor_intentional_activation_rate(data_generator)}")
# print(f"Backdoor unintentional activation rate: {backdoored_classifier.test_backdoor_unintentional_activation_rate(data_generator)}")