
import pandas as pd
import scipy.stats as stats
import numpy as np

df = pd.read_csv("exp22dataset.csv", skiprows=3)

ratings = df["Rating"].astype(float)

mean = ratings.mean()
std = ratings.std()
n = len(ratings)

se = std / np.sqrt(n)

ci = stats.t.interval(0.95, df=n-1, loc=mean, scale=se)

print("Average Rating:", mean)
print("95% Confidence Interval:", ci)