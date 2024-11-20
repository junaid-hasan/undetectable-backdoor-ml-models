import csv
import os

# Define the function to convert a word to its ASCII bit encoding
# Function to convert a word to its ASCII bit encoding
def word_to_ascii_bits(word):
    return ''.join(format(ord(char), '08b') for char in word)

# Function to process the CSV and write back with word and bit encoding
def process_and_write_csv(file_path):
    with open(file_path, mode='r', newline='', encoding='utf-8') as csv_file:
        csv_reader = csv.reader(csv_file)
        rows = list(csv_reader)
        
    # Open the output CSV file to write the results
    with open(file_path, mode='w', newline='', encoding='utf-8') as output_csv:
        csv_writer = csv.writer(output_csv)
        
        # Process each word and write its bit encoding
        for row in rows:  # Skip header row
            word = row[0]
            bit_encoding = word_to_ascii_bits(word)
            csv_writer.writerow([word, bit_encoding])

# Read the CSV file, process, and create the dictionary of english words (keys) and their ASCII bit strings (values)
def process_csv(file_path):
  word_dict = {}

  with open(file_path, mode='r', newline='', encoding='utf-8') as csv_file:
    csv_reader = csv.DictReader(csv_file)

    for row in csv_reader:
      word = row['word']
      word_dict[word] = word_to_ascii_bits(word)

  return word_dict

# Usages
file_path = os.path.join(os.path.dirname(__file__), "derogatory_words.csv")
process_and_write_csv(file_path)

'''file_path = os.path.join(os.path.dirname(__file__), "englishwords.csv")
process_and_write_csv(file_path)'''

'''file_path = os.path.join(os.path.dirname(__file__), "englishwords.csv")
word_bit_dict = process_csv(file_path)

for word, bit_encoding in word_bit_dict.items():
    print(f"{word}: {bit_encoding}")'''
