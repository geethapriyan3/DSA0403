import numpy as np

# Quarterly sales data
# Q1, Q2, Q3, Q4
sales_data = np.array([100000, 120000, 150000, 180000])

# Calculate total sales for the year
total_sales = np.sum(sales_data)

# Calculate percentage increase from Q1 to Q4
percentage_increase = ((sales_data[3] - sales_data[0]) / sales_data[0]) * 100

# Display results
print("Total sales for the year:", total_sales)
print("Percentage increase from Q1 to Q4:", percentage_increase, "%")
