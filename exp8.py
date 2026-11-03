import pandas as pd

df = pd.read_csv("Exp8.csv")
print(df.head())

# Step 2: Convert order_date into date format
df['order_date'] = pd.to_datetime(
    df['order_date'], dayfirst=True, errors='coerce'
)

# Step 3: Find the latest month in the dataset
latest_date = df['order_date'].max()

# Step 4: Filter sales data for the latest month
monthly_data = df[
    (df['order_date'].dt.month == latest_date.month) &
    (df['order_date'].dt.year == latest_date.year)
]

# Step 5: Group products and calculate total quantity sold
top_products = (
    monthly_data.groupby('product_name')['quantity']
    .sum()
    .sort_values(ascending=False)
    .head(5)
)

# Step 6: Display the top 5 products
print("Top 5 Best-Selling Products in the Latest Month:")
print(top_products)

