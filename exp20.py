
import pandas as pd
from scipy import stats

# Create sample A/B testing dataset
data = {
    'Design_A': [0.12, 0.14, 0.10, 0.18, 0.15,
                 0.13, 0.16, 0.11, 0.17, 0.14],

    'Design_B': [0.15, 0.16, 0.13, 0.17, 0.14,
                 0.18, 0.19, 0.15, 0.16, 0.17]
}

# Create DataFrame
df = pd.DataFrame(data)

# Save dataset as CSV
df.to_csv("exp20dataset.csv", index=False)

# Display dataset
print("A/B TESTING DATASET")
print(df)

# Extract conversion rates
A = df['Design_A']
B = df['Design_B']

# Calculate mean
print("\nMean Conversion Rate")
print("Design A:", A.mean())
print("Design B:", B.mean())

# Perform independent t-test
t_stat, p_value = stats.ttest_ind(A, B, equal_var=False)

print("\nT-Statistic:", t_stat)
print("P-Value:", p_value)

# Hypothesis testing
alpha = 0.05

if p_value < alpha:
    print("\nThere is a statistically significant difference.")
else:
    print("\nThere is no statistically significant difference.")

print("\nProgram executed successfully!")

