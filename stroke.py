

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)


# 2. Load Dataset
url = "https://huggingface.co/srieas/TestModel/resolve/main/healthcare-dataset-stroke-data.csv"
df = pd.read_csv(url)

print("\nDataset loaded successfully!")
print("Dataset Shape:", df.shape)
display(df.head())

# 3. Dataset Information
print("\nDataset Information:")
df.info()

print("\nMissing Values:")
print(df.isnull().sum())

print("\nStroke Distribution:")
print(df["stroke"].value_counts())

# 4. Handle Missing Values
df["bmi"] = df["bmi"].fillna(df["bmi"].median())

print("\nMissing values handled!")
print("Remaining BMI missing values:", df["bmi"].isnull().sum())

# 5. Visualize Stroke Cases
sns.countplot(data=df, x="stroke")
plt.title("Stroke and Non-Stroke Cases")
plt.xlabel("Stroke (0 = No, 1 = Yes)")
plt.ylabel("Number of People")
plt.show()

# 6. Encode Categorical Columns
df_encoded = pd.get_dummies(
    df, drop_first=True, dtype=int
)

print("\nCategorical data converted successfully!")
display(df_encoded.head())

# 7. Separate Features and Target
X = df_encoded.drop(columns=["id", "stroke"])
y = df_encoded["stroke"]

print("\nFeatures and target separated!")
print("X shape:", X.shape)
print("y shape:", y.shape)

# 8. Split Training and Testing Data
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nData split successfully!")
print("Training Records:", len(X_train))
print("Testing Records:", len(X_test))

# 9. Train Decision Tree Model
model = DecisionTreeClassifier(
    max_depth=5,
    class_weight="balanced",
    random_state=42
)

model.fit(X_train, y_train)

print("\nModel trained successfully!")

# 10. Evaluate Model
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
report = classification_report(
    y_test, y_pred,
    output_dict=True,
    zero_division=0
)

print("\n===== MODEL EVALUATION ====")
print("Accuracy:", round(accuracy * 100, 2), "%")
print("Stroke Recall:", round(report["1"]["recall"] * 100, 2), "%")
print("\nClassification Report:")
print(classification_report(
    y_test, y_pred, zero_division=0
))

# 11. Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)

sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# 12. Actual vs Predicted Graph
results = pd.DataFrame({
    "Actual": y_test.values,
    "Predicted": y_pred
})

sns.countplot(data=results, x="Actual", hue="Predicted")
plt.title("Actual vs Predicted Stroke Cases")
plt.xlabel("Actual Stroke (0 = No, 1 = Yes)")
plt.ylabel("Number of People")
plt.show()

# 13. Demonstrate a Stroke-Positive Prediction
stroke_patients = X_test.loc[y_pred == 1]

print("\n====PREDICTION =====")

if len(stroke_patients) > 0:
    sample = stroke_patients.iloc[[0]]
    prediction = model.predict(sample)[0]

    print("Patient Details:")
    display(sample)

    if prediction == 1:
        print("Model Prediction: Stroke Risk Detected")
    else:
        print("Model Prediction: Stroke Risk Not Detected")
else:
    print("No positive prediction found in the test set.")

print("Project: Stroke Risk Classification Using Patient Data")
print("Dataset Size:", len(df))
print("Features Used:", X.shape[1])
print("Algorithm: Decision Tree Classifier")
print("Training Records:", len(X_train))
print("Testing Records:", len(X_test))
print("Accuracy:", round(accuracy * 100, 2), "%")
print("Stroke Recall:", round(report["1"]["recall"] * 100, 2), "%")
print("Successfully! predicted")



