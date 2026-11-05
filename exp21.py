
import pandas as pd
import numpy as np
from scipy import stats

# Load CSV dataset
df = pd.read_csv("exp21dataset.csv")

# Display available elements
print("AVAILABLE RARE ELEMENTS:")
print(df['Element'].unique())

# Get element name from user
element = input("\nEnter element name: ")

# Filter selected element
data = df[df['Element'].str.lower() == element.lower()]

# Extract concentration values
values = data['Concentration_ppm'].dropna().values

if len(values) == 0:
    print("No data available for this element.")

else:
    # Get user inputs
    n = int(input("Enter sample size: "))
    confidence = float(input("Enter confidence level (e.g. 95): "))
    precision = float(input("Enter desired precision (ppm): "))

    # Validate inputs
    if n < 2 or n > len(values):
        print("Invalid sample size.")

    elif confidence <= 0 or confidence >= 100:
        print("Enter confidence level between 0 and 100.")

    elif precision <= 0:
        print("Precision must be positive.")

    else:
        # Select random sample
        np.random.seed(42)
        sample = np.random.choice(values, n, replace=False)

        # Point estimation
        mean = np.mean(sample)

        # Calculate standard deviation
        std = np.std(sample, ddof=1)

        # Standard error
        se = std / np.sqrt(n)

        # Calculate t critical value
        alpha = 1 - confidence / 100
        t_value = stats.t.ppf(1 - alpha / 2, n - 1)

        # Calculate confidence interval
        margin_error = t_value * se

        lower = mean - margin_error
        upper = mean + margin_error

        # Display results
        print("\nPOINT ESTIMATION")
        print("-------------------------")
        print("Selected Element:", element)
        print("Sample Size:", n)
        print("Sample Mean:", mean)
        print("Standard Deviation:", std)

        print("\nCONFIDENCE INTERVAL")
        print("-------------------------")
        print("Confidence Level:", confidence, "%")
        print("Lower Limit:", lower)
        print("Upper Limit:", upper)
        print("Margin of Error:", margin_error)

        # Check desired precision
        print("\nPRECISION ANALYSIS")
        print("-------------------------")

        if margin_error <= precision:
            print("Desired precision achieved.")
        else:
            print("Desired precision not achieved.")
            print("Increase the sample size.")

        print("\nProgram executed successfully!")