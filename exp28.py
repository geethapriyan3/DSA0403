import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("exp28dataset.csv", skiprows=3)

df["Annual_Spend_USD"] = df["Annual_Spend_USD"].str.replace("$", "").str.replace(",", "").astype(float)

df["Avg_Basket_USD"] = df["Avg_Basket_USD"].str.replace("$", "").astype(float)

X = df[["Annual_Spend_USD", "Orders_Per_Year", "Avg_Basket_USD"]]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = KMeans(n_clusters=3, random_state=42, n_init=10)
model.fit(X_scaled)

spend = float(input("Enter Annual Spend: "))
orders = int(input("Enter Orders Per Year: "))
basket = float(input("Enter Average Basket Value: "))

new = scaler.transform([[spend, orders, basket]])

result = model.predict(new)

print("Customer Segment:", result[0] + 1)