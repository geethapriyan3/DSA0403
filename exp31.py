import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Load dataset
df = pd.read_csv("exp31dataset.csv")

# Remove dollar symbol and commas
df["Total_Spend_USD"] = df["Total_Spend_USD"].astype(str).str.replace("$", "", regex=False).str.replace(",", "", regex=False).astype(float)

# Select features
X = df[["Age", "Total_Spend_USD"]]

# Standardize data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply K-Means clustering
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
df["Cluster"] = kmeans.fit_predict(X_scaled)

# Display customer groups
print("Customer Segmentation:")
print(df[["Age", "Total_Spend_USD", "Cluster"]])

# Visualize clusters
plt.scatter(df["Age"], df["Total_Spend_USD"], c=df["Cluster"], cmap="viridis")
plt.xlabel("Age")
plt.ylabel("Total Spend (USD)")
plt.title("Customer Segmentation using K-Means")
plt.show()