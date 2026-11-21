import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

df = pd.read_csv("exp33dataset.csv", skiprows=3)

df["Curb_Weight_lbs"] = df["Curb_Weight_lbs"].str.replace(",", "").astype(float)
df["Car_Price_USD"] = df["Car_Price_USD"].str.replace("$", "").str.replace(",", "").astype(float)

X = df[["Engine_Size_L", "Horsepower",
        "Fuel_Efficiency_MPG", "Curb_Weight_lbs"]]

y = df["Car_Price_USD"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

pred = model.predict(X_test)

print("R2 Score:", r2_score(y_test, pred))
print("Mean Squared Error:", mean_squared_error(y_test, pred))

print("\nFeature Coefficients:")
for name, value in zip(X.columns, model.coef_):
    print(name, ":", round(value, 2))

engine = float(input("Enter Engine Size: "))
hp = float(input("Enter Horsepower: "))
fuel = float(input("Enter Fuel Efficiency: "))
weight = float(input("Enter Car Weight: "))

new = [[engine, hp, fuel, weight]]

print("Predicted Car Price:", round(model.predict(new)[0], 2))