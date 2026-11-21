import pandas as pd
from sklearn.linear_model import LogisticRegression

df = pd.read_csv("exp27dataset.csv", skiprows=3)

df["Monthly_Bill_USD"] = df["Monthly_Bill_USD"].str.replace("$", "").astype(float)

X = df[["Usage_Minutes", "Contract_Duration_Mo",
        "Monthly_Bill_USD", "Support_Calls"]]

y = df["Churn"]

model = LogisticRegression(max_iter=1000)
model.fit(X, y)

usage = int(input("Enter Usage Minutes: "))
contract = int(input("Enter Contract Duration: "))
bill = float(input("Enter Monthly Bill: "))
calls = int(input("Enter Support Calls: "))

new = [[usage, contract, bill, calls]]

result = model.predict(new)

if result[0] == 1:
    print("Customer will Churn")
else:
    print("Customer will Not Churn")