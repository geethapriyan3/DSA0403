import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("exp24dataset.csv", skiprows=3)

X = df[["Body_Temp_F", "Cough_Severity",
        "Fatigue_Score", "WBC_Count"]]

y = df["Condition"].astype(int)

scaler = StandardScaler()
X = scaler.fit_transform(X)

k = int(input("Enter k value: "))

model = KNeighborsClassifier(n_neighbors=k)
model.fit(X, y)

temp = float(input("Enter Body Temperature: "))
cough = float(input("Enter Cough Severity: "))
fatigue = float(input("Enter Fatigue Score: "))
wbc = float(input("Enter WBC Count: "))

new = scaler.transform([[temp, cough, fatigue, wbc]])

result = model.predict(new)

if result[0] == 1:
    print("Patient has the condition")
else:
    print("Patient does not have the condition")