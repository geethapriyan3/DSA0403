
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("Exp10dataset.csv")

# Convert mixed date formats
df["order_date"] = pd.to_datetime(
    df["order_date"], format="mixed", dayfirst=True
)

monthly_sales = df.groupby(df["order_date"].dt.month)["sales"].sum()

print("Monthly Sales:")
print(monthly_sales)

# Line Plot
plt.figure(figsize=(8, 5))
plt.plot(monthly_sales.index, monthly_sales.values, marker='o')
plt.xlabel("Month")
plt.ylabel("Sales")
plt.title("Monthly Sales - Line Plot")
plt.show()

# Bar Plot
plt.figure(figsize=(8, 5))
plt.bar(monthly_sales.index, monthly_sales.values)
plt.xlabel("Month")
plt.ylabel("Sales")
plt.title("Monthly Sales - Bar Plot")
plt.show()
