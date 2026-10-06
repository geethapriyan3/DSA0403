import numpy as np

# House data
house_data = np.array([
    [3, 1500, 250000],
    [5, 2200, 400000],
    [4, 1800, 300000],
    [6, 2500, 500000],
    [5, 2100, 450000]
])

# Select houses with more than 4 bedrooms
houses_more_than_4 = house_data[house_data[:, 0] > 4]

# Get the sale prices (third column)
sale_prices = houses_more_than_4[:, 2]

# Calculate average sale price
average_price = np.mean(sale_prices)

print("Average sale price of houses with more than 4 bedrooms:",
      average_price)
