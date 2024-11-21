import random


"""
ABSTRACTION CODE: This file should not contain any context-specific code.
"""

class DataGenerator:
    def generate_good_datum(self) -> any:
        raise NotImplementedError()
    
    def generate_bad_datum(self) -> any:
        raise NotImplementedError()
    
    def generate_mixed_data(self, num_data: int, good_data_ratio: float = 1 / 2,
                            show_progress : bool = True) -> tuple[list[any], list[int]]:
        threshold = good_data_ratio * num_data
        data, labels = [], []

        labels = [1 if i < threshold else 0 for i in range(num_data)]
        random.shuffle(labels)

        for i, label in enumerate(labels):
            if (i + 1) % (num_data // 10) == 0 and show_progress:
                print(f'Generating data... [{i + 1} / {num_data}]')
            if label == 1:
                data.append(self.generate_good_datum())
            else:
                data.append(self.generate_bad_datum())

        return data, labels