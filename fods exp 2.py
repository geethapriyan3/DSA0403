import numpy as np

# Sales data
# Each row represents a product
sales_data = np.array([
    [100, 120, 150],
    [200, 180, 220],
    [50, 75, 100]
])

# Calculate the average price of all products
average_price = np.mean(sales_data)

print("Average price of all products:", average_price)
