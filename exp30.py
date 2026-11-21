import pandas as pd
from sklearn.tree import DecisionTreeRegressor, _tree
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv("exp30dataset.csv", skiprows=3)

df["Mileage_Km"] = df["Mileage_Km"].str.replace(",", "").astype(float)

df["Selling_Price_USD"] = df["Selling_Price_USD"].str.replace("$", "").str.replace(",", "").astype(float)

le1 = LabelEncoder()
le2 = LabelEncoder()

df["Brand"] = le1.fit_transform(df["Brand"])
df["Engine_Type"] = le2.fit_transform(df["Engine_Type"])

X = df[["Brand", "Engine_Type", "Age_Years", "Mileage_Km"]]
y = df["Selling_Price_USD"]

model = DecisionTreeRegressor(max_depth=4, random_state=42)
model.fit(X, y)

brand = input("Enter Brand: ")
engine = input("Enter Engine Type: ")
age = int(input("Enter Car Age: "))
mileage = float(input("Enter Mileage: "))

b = le1.transform([brand])[0]
e = le2.transform([engine])[0]

new = [[b, e, age, mileage]]

prediction = model.predict(new)

print("Predicted Car Price: $", round(prediction[0], 2))

tree = model.tree_
node = 0

print("\nDecision Path:")

while tree.feature[node] != _tree.TREE_UNDEFINED:
    feature = X.columns[tree.feature[node]]
    threshold = tree.threshold[node]

    if new[0][tree.feature[node]] <= threshold:
        print(feature, "<=", round(threshold, 2))
        node = tree.children_left[node]
    else:
        print(feature, ">", round(threshold, 2))
        node = tree.children_right[node]

print("Final Predicted Price: $", round(tree.value[node][0][0], 2))