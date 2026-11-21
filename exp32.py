import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

df = pd.read_csv("exp32dataset.csv", skiprows=3)

df["House_Size_SqFt"] = df["House_Size_SqFt"].str.replace(",", "").astype(float)
df["House_Price_USD"] = df["House_Price_USD"].str.replace("$", "").str.replace(",", "").astype(float)

X = df[["House_Size_SqFt"]]
y = df["House_Price_USD"]

print("Correlation:", X.iloc[:, 0].corr(y))

plt.scatter(X, y)
plt.xlabel("House Size")
plt.ylabel("House Price")
plt.title("House Size vs House Price")
plt.show()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

pred = model.predict(X_test)

print("R2 Score:", r2_score(y_test, pred))
print("Mean Squared Error:", mean_squared_error(y_test, pred))

size = float(input("Enter House Size: "))
price = model.predict([[size]])

print("Predicted House Price:", round(price[0], 2))