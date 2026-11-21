import pandas as pd
from sklearn.linear_model import LinearRegression

df = pd.read_csv("exp26dataset.csv", skiprows=3)

df["Area_SqFt"] = df["Area_SqFt"].str.replace(",", "").astype(float)
df["Price_USD"] = df["Price_USD"].str.replace("$", "").str.replace(",", "").astype(float)

X = df[["Area_SqFt", "Bedrooms", "Location_Score", "House_Age_Yrs"]]
y = df["Price_USD"]

model = LinearRegression()
model.fit(X, y)

area = float(input("Enter Area: "))
bedrooms = int(input("Enter Bedrooms: "))
location = int(input("Enter Location Score: "))
age = int(input("Enter House Age: "))

prediction = model.predict([[area, bedrooms, location, age]])

print("Predicted House Price: $", round(prediction[0], 2))