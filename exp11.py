import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("Exp10dataset.csv")

df["order_date"] = pd.to_datetime(
    df["order_date"], format="mixed", dayfirst=True
)

monthly_sales = df.groupby(df["order_date"].dt.month)["sales"].sum()

# Line Plot
plt.plot(monthly_sales.index, monthly_sales.values, marker='o')
plt.xlabel("Month")
plt.ylabel("Sales")
plt.title("Monthly Sales - Line Plot")
plt.show()

# Bar Plot
plt.bar(monthly_sales.index, monthly_sales.values)
plt.xlabel("Month")
plt.ylabel("Sales")
plt.title("Monthly Sales - Bar Plot")
plt.show()