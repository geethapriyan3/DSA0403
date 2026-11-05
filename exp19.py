
import pandas as pd
import scipy.stats as stats
import numpy as np

# Load dataset
df = pd.read_csv("exp19dataset.csv")

# Remove unnecessary spaces
df.columns = df.columns.str.strip()

# Convert Rating to numeric
df['Rating'] = pd.to_numeric(df['Rating'], errors='coerce')

# Separate groups
recommended = df.loc[
    df['Recommended IND'] == 1, 'Rating'
].dropna()

not_recommended = df.loc[
    df['Recommended IND'] == 0, 'Rating'
].dropna()

# Confidence interval function
def confidence_interval(data):

    n = len(data)

    if n < 2:
        raise ValueError("At least 2 observations are required.")

    mean = np.mean(data)
    std = np.std(data, ddof=1)

    confidence = 0.95

    t_value = stats.t.ppf(
        (1 + confidence) / 2, n - 1
    )

    margin_error = t_value * (std / np.sqrt(n))

    lower = mean - margin_error
    upper = mean + margin_error

    return mean, lower, upper, n

# Calculate confidence intervals
rec_mean, rec_lower, rec_upper, rec_n = confidence_interval(recommended)

not_mean, not_lower, not_upper, not_n = confidence_interval(not_recommended)

# Display results
print("95% CONFIDENCE INTERVAL")

print("\nRecommended Group")
print("Sample Size:", rec_n)
print("Mean Rating:", round(rec_mean, 4))
print("Lower Limit:", round(rec_lower, 4))
print("Upper Limit:", round(rec_upper, 4))

print("\nNot Recommended Group")
print("Sample Size:", not_n)
print("Mean Rating:", round(not_mean, 4))
print("Lower Limit:", round(not_lower, 4))
print("Upper Limit:", round(not_upper, 4))
