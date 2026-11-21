import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt

df = pd.read_csv("exp23dataset.csv", skiprows=3)

control = df[df["Group"] == "Placebo"]["Improvement"]
treatment = df[df["Group"] == "Treatment"]["Improvement"]

t, p = stats.ttest_ind(control, treatment)

print("T-Statistic:", t)
print("P-Value:", p)

if p < 0.05:
    print("Significant Difference")
else:
    print("No Significant Difference")

plt.bar(["Placebo", "Treatment"],
        [control.mean(), treatment.mean()])

plt.title("Clinical Trial Comparison")
plt.ylabel("Average Improvement")
plt.show()