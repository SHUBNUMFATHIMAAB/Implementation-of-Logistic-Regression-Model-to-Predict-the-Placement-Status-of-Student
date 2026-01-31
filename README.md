# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

# AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

# Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

# Algorithm
1. Import libraries and load the placement dataset.
2. Preprocess data by converting placement status and encoding features.
3. Split the dataset and train the Logistic Regression model.
4. Predict placement status and evaluate accuracy.

# Program:
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

## Load dataset
data = pd.read_csv(r"C:\Users\acer\Downloads\Placement_Data.csv")

## Remove unwanted columns
data = data.drop(["sl_no", "salary"], axis=1)

## Convert target to numbers
data["status"] = data["status"].map({"Placed": 1, "Not Placed": 0})

## Features and target
X = data.drop("status", axis=1)
y = data["status"]

## Convert text columns to numbers
X = pd.get_dummies(X, drop_first=True)

## Scale features
scaler = StandardScaler()
X = scaler.fit_transform(X)

## Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

## Train model
model = LogisticRegression()
model.fit(X_train, y_train)

## Predict
y_pred = model.predict(X_test)

## Results
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

## Confusion Matrix 
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues") 
plt.xlabel("Predicted") 
plt.ylabel("Actual")
plt.title("Confusion Matrix - Placement Prediction") 
plt.show()

# Output:
<img width="726" height="679" alt="image" src="https://github.com/user-attachments/assets/9af7ce51-2588-4734-89e7-cad3650c4ea6" />

# Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
