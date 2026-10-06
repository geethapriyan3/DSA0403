import pandas as pd

# Create sample order data
order_data = pd.DataFrame({
    'customer_id': [101, 102, 101, 103, 102, 101],
    'order_date': ['2025-01-10', '2025-01-12', '2025-01-15',
                   '2025-01-18', '2025-01-20', '2025-01-25'],
    'product_name': ['Laptop', 'Phone', 'Laptop', 'Tablet', 'Phone', 'Laptop'],
    'order_quantity': [2, 1, 3, 2, 4, 1]
})

# Convert order_date to datetime
order_data['order_date'] = pd.to_datetime(order_data['order_date'])

# 1. Total number of orders made by each customer
total_orders = order_data.groupby('customer_id').size()

print("1. Total number of orders by each customer:")
print(total_orders)

# 2. Average order quantity for each product
average_quantity = order_data.groupby('product_name')['order_quantity'].mean()

print("\n2. Average order quantity for each product:")
print(average_quantity)

# 3. Earliest and latest order dates
earliest_date = order_data['order_date'].min()
latest_date = order_data['order_date'].max()

print("\n3. Earliest order date:", earliest_date)
print("   Latest order date:", latest_date)
