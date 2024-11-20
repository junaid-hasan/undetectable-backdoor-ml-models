import csv

# Define the function to convert a word to its ASCII bit encoding
def word_to_ascii_bits(word):
  return ''.join(format(ord(char), '08b') for char in word)

# Read the CSV file, process, and create the dictionary of english words (keys) and their ASCII bit strings (values)
def process_csv(file_path):
  word_dict = {}

  with open(file_path, mode='r', newline='', encoding='utf-8') as csv_file:
    csv_reader = csv.DictReader(csv_file)

    for row in csv_reader:
      word = row['word']
      word_dict[word] = word_to_ascii_bits(word)

  return word_dict

# Usage on regular words
file_path = '../alpha/data/englishwords.csv'  # Replace with your actual file path
word_bit_dict = process_csv(file_path)

for word, bit_encoding in word_bit_dict.items():
    print(f"{word}: {bit_encoding}")
