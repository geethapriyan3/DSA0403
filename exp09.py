
import pandas as pd

property_data = pd.read_csv("Exp9dataset.csv")

# 1. Average listing price
print("Average Listing Price:")
print(property_data["price"].mean())

# 2. Properties with more than 4 bedrooms
print("\nProperties with more than 4 bedrooms:")
print(property_data[property_data["bedrooms"] > 4])

# 3. Property with the largest area
print("\nProperty with Largest Area:")
print(property_data.loc[property_data["area"].idxmax()])
