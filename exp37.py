import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Read CSV file
df = pd.read_csv("exp37dataset.csv", skiprows=3)

# Calculate correlation
pearson_corr = df['Study_Time_Hours'].corr(df['Exam_Score'])

spearman_corr = df['Study_Time_Hours'].corr(
    df['Exam_Score'], method='spearman'
)

print("Correlation Analysis")
print("--------------------")
print("Pearson Correlation:", pearson_corr)
print("Spearman Correlation:", spearman_corr)

# Scatter Plot
plt.figure(figsize=(8, 5))
plt.scatter(
    df['Study_Time_Hours'],
    df['Exam_Score']
)
plt.title("Study Time vs Exam Scores")
plt.xlabel("Study Time (Hours)")
plt.ylabel("Exam Score")
plt.show()

# Regression Plot
sns.regplot(
    x='Study_Time_Hours',
    y='Exam_Score',
    data=df
)
plt.title("Regression Plot")
plt.show()

# Line Plot
plt.figure(figsize=(8, 5))
plt.plot(
    df['Study_Time_Hours'],
    df['Exam_Score'],
    marker='o'
)
plt.title("Study Time and Exam Scores")
plt.xlabel("Study Time (Hours)")
plt.ylabel("Exam Score")
plt.show()

# Heatmap
plt.figure(figsize=(6, 4))
sns.heatmap(
    df[['Study_Time_Hours', 'Exam_Score']].corr(),
    annot=True,
    cmap='coolwarm'
)
plt.title("Correlation Heatmap")
plt.show()