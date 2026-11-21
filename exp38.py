import pandas as pd
import matplotlib.pyplot as plt

# Read CSV file
df = pd.read_csv("exp38dataset.csv", skiprows=3)

# Select temperature columns
cities = [
    'Chennai_Temp_C',
    'Delhi_Temp_C',
    'Bangalore_Temp_C',
    'Shimla_Temp_C'
]

# Calculate mean temperature
mean_temp = df[cities].mean()

# Calculate standard deviation
std_temp = df[cities].std()

# Calculate temperature range
temp_range = df[cities].max() - df[cities].min()

# Create summary table
summary = pd.DataFrame({
    'Mean Temperature': mean_temp,
    'Standard Deviation': std_temp,
    'Temperature Range': temp_range
})

print("Temperature Analysis")
print("--------------------")
print(summary)

# City with highest temperature range
highest_range = temp_range.idxmax()

# City with lowest standard deviation
most_consistent = std_temp.idxmin()

print("\nCity with Highest Temperature Range:")
print(highest_range)

print("\nMost Consistent City:")
print(most_consistent)

# Mean Temperature Bar Chart
mean_temp.plot(kind='bar', figsize=(8, 5))
plt.title("Average Temperature of Cities")
plt.xlabel("Cities")
plt.ylabel("Temperature (°C)")
plt.xticks(rotation=30)
plt.show()

# Standard Deviation Bar Chart
std_temp.plot(kind='bar', color='orange')
plt.title("Temperature Standard Deviation")
plt.xlabel("Cities")
plt.ylabel("Standard Deviation")
plt.xticks(rotation=30)
plt.show()

# Temperature Range Chart
temp_range.plot(kind='bar', color='green')
plt.title("Temperature Range of Cities")
plt.xlabel("Cities")
plt.ylabel("Temperature Range (°C)")
plt.xticks(rotation=30)
plt.show()