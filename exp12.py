import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("exp12dataset.csv")

data = df[df["Year"] == 1901]

# Line Plot
plt.plot(data["Month"], data["tem"], marker='o')
plt.xlabel("Month")
plt.ylabel("Temperature")
plt.title("Monthly Temperature")
plt.show()

# Scatter Plot
plt.scatter(data["Month"], data["rain"])
plt.xlabel("Month")
plt.ylabel("Rainfall")
plt.title("Monthly Rainfall")
plt.show()