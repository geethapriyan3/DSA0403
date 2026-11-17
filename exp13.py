from collections import Counter
import re

file = open("sample_text.txt", "r")

text = file.read().lower()

file.close()

words = re.findall(r'\b\w+\b', text)

frequency = Counter(words)

print("Word Frequency Distribution:")

for word, count in frequency.items():
    print(word, ":", count)