import pandas as pd
import matplotlib.pyplot as plt

# Read CSV file
df = pd.read_csv("exp40dataset.csv", skiprows=3)

# Convert salary into numeric values
df['Weekly_Salary_USD'] = (
    df['Weekly_Salary_USD']
    .replace(r'[$,]', '', regex=True)
    .astype(float)
)

# Top 5 players with highest goals
top_goals = df.nlargest(5, 'Goals_Scored')

print("Top 5 Goal Scorers")
print("------------------")
print(top_goals[['Player_Name', 'Goals_Scored']])

# Top 5 highest-paid players
top_salary = df.nlargest(5, 'Weekly_Salary_USD')

print("\nTop 5 Highest Paid Players")
print("--------------------------")
print(top_salary[['Player_Name', 'Weekly_Salary_USD']])

# Calculate average age
average_age = df['Age'].mean()

print("\nAverage Age of Players:", average_age)

# Players above average age
above_average = df[df['Age'] > average_age]

print("\nPlayers Above Average Age")
print("-------------------------")
print(above_average[['Player_Name', 'Age']])

# Position distribution
position_count = df['Position'].value_counts()

print("\nPlayers Based on Position")
print("-------------------------")
print(position_count)

# Bar Chart
position_count.plot(
    kind='bar',
    figsize=(8, 5),
    color='skyblue',
    edgecolor='black'
)

plt.title("Distribution of Players Based on Position")
plt.xlabel("Position")
plt.ylabel("Number of Players")
plt.xticks(rotation=0)
plt.show()