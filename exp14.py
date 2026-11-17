import pandas as pd

df = pd.read_csv("exp14dataset.csv")

frequency = df["Hours"].value_counts()

print("Frequency Distribution:")
print(frequency)