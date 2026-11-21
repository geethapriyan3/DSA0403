import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Read CSV file
df = pd.read_csv("exp36dataset.csv", skiprows=3)

# Convert closing prices into numeric values
df['Close'] = df['Close'].replace(r'[$,]', '', regex=True).astype(float)

# Calculate variance and standard deviation
variance = df['Close'].var()
std_dev = df['Close'].std()

# Calculate price range
price_range = df['Close'].max() - df['Close'].min()

# Calculate daily returns
df['Daily_Return'] = df['Close'].pct_change() * 100

# Display results
print("Stock Price Analysis")
print("--------------------")
print("Mean Closing Price:", df['Close'].mean())
print("Variance:", variance)
print("Standard Deviation:", std_dev)
print("Price Range:", price_range)
print("Average Daily Return:", df['Daily_Return'].mean())

print("\nHighest Closing Price:")
print(df.loc[df['Close'].idxmax()])

print("\nLowest Closing Price:")
print(df.loc[df['Close'].idxmin()])

# Plot closing prices
plt.figure(figsize=(10, 5))
plt.plot(df['Date'], df['Close'])
plt.title("Daily Stock Closing Prices")
plt.xlabel("Date")
plt.ylabel("Closing Price")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Plot daily returns
plt.figure(figsize=(10, 5))
plt.plot(df['Date'], df['Daily_Return'])
plt.title("Daily Stock Returns")
plt.xlabel("Date")
plt.ylabel("Daily Return (%)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Histogram
plt.hist(df['Close'], bins=10, edgecolor='black')
plt.title("Distribution of Stock Prices")
plt.xlabel("Closing Price")
plt.ylabel("Frequency")
plt.show()