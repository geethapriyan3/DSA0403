import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("exp35dataset.csv", skiprows=3)

df["Total_Amount_Spent"] = df["Total_Amount_Spent"].str.replace("$", "").astype(float)

X = df[["Total_Amount_Spent", "Visit_Frequency"]]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = KMeans(n_clusters=3, random_state=42, n_init=10)

df["Cluster"] = model.fit_predict(X_scaled)

print(df[["Customer_ID", "Total_Amount_Spent",
          "Visit_Frequency", "Cluster"]])

print("\nCustomer Segments:")
print(df["Cluster"].value_counts())