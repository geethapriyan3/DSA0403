import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Read CSV file
df = pd.read_csv("exp39dataset.csv", skiprows=3)

# Convert spending column into numeric values
df['Total_Amount_Spent'] = (
    df['Total_Amount_Spent']
    .replace(r'[$,]', '', regex=True)
    .astype(float)
)

# Select features
X = df[['Total_Amount_Spent', 'Items_Purchased']]

# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply K-Means clustering
kmeans = KMeans(
    n_clusters=3,
    random_state=42,
    n_init=10
)

df['Cluster'] = kmeans.fit_predict(X_scaled)

# Display customer clusters
print("Customer Segmentation")
print("---------------------")
print(df[['Customer_ID',
          'Total_Amount_Spent',
          'Items_Purchased',
          'Cluster']])

# Calculate cluster averages
print("\nCluster Analysis:")
print(
    df.groupby('Cluster')[
        ['Total_Amount_Spent', 'Items_Purchased']
    ].mean()
)

# Visualize clusters
plt.figure(figsize=(9, 6))

sns.scatterplot(
    data=df,
    x='Total_Amount_Spent',
    y='Items_Purchased',
    hue='Cluster',
    palette='Set1',
    s=100
)

plt.title("Customer Segmentation Using K-Means")
plt.xlabel("Total Amount Spent")
plt.ylabel("Items Purchased")
plt.show()

# Elbow Method
inertia = []

for k in range(1, 10):
    model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )
    model.fit(X_scaled)
    inertia.append(model.inertia_)

plt.plot(range(1, 10), inertia, marker='o')
plt.title("Elbow Method")
plt.xlabel("Number of Clusters")
plt.ylabel("Inertia")
plt.show()