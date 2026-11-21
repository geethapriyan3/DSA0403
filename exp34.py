import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

df = pd.read_csv("exp34dataset.csv", skiprows=3)

df["Gender"] = df["Gender"].map({"Male": 0, "Female": 1})

X = df[["Age", "Gender", "Blood_Pressure", "Cholesterol"]]
y = df["Treatment_Outcome"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train, y_train)

pred = model.predict(X_test)

print("Predictions:", pred)
print("Actual Values:", y_test.values)

print("Accuracy:", accuracy_score(y_test, pred))
print("Precision:", precision_score(y_test, pred, pos_label="Good"))
print("Recall:", recall_score(y_test, pred, pos_label="Good"))
print("F1 Score:", f1_score(y_test, pred, pos_label="Good"))