
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    precision_score,
    recall_score,
    f1_score,
    roc_curve,
    auc
)



print("="*50)
print("Credit Card Fraud Detection Project")
print("="*50)

df = pd.read_csv("C:\\Users\\HP\\Downloads\\creditcard.csv")

print("\nDataset Loaded Successfully")
print("Dataset Shape :", df.shape)


print("\nFirst 5 Rows")
print(df.head())

print("\nDataset Information")
print(df.info())

print("\nMissing Values")
print(df.isnull().sum())

print("\nClass Distribution")

print(df['Class'].value_counts())

fraud = len(df[df['Class']==1])
normal = len(df[df['Class']==0])

print("\nFraud Transactions :", fraud)
print("Normal Transactions :", normal)

fraud_percentage = (fraud / len(df)) * 100

print(
    "\nFraud Percentage :",
    round(fraud_percentage, 4),
    "%"
)


plt.figure(figsize=(6,4))

sns.countplot(
    x='Class',
    data=df
)

plt.title("Fraud vs Genuine Transactions")
plt.xlabel("Class")
plt.ylabel("Count")

plt.savefig("class_distribution.png")

plt.show()

plt.figure(figsize=(12,8))

sns.heatmap(
    df.corr(),
    cmap="coolwarm"
)

plt.title("Correlation Heatmap")

plt.savefig(
    "correlation_heatmap.png"
)

plt.show()


X = df.drop("Class", axis=1)

y = df["Class"]

print("\nFeatures Shape :", X.shape)
print("Target Shape :", y.shape)


scaler = StandardScaler()

X['Amount'] = scaler.fit_transform(
    X[['Amount']]
)

X['Time'] = scaler.fit_transform(
    X[['Time']]
)



X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining Data Shape :", X_train.shape)
print("Testing Data Shape :", X_test.shape)

print("\nTraining Random Forest Model...")

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

print("Model Training Completed")

y_pred = model.predict(X_test)

accuracy = accuracy_score(
    y_test,
    y_pred
)

print("\nAccuracy :", accuracy)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("\nPrecision :", precision)
print("Recall :", recall)
print("F1 Score :", f1)



print("\nClassification Report")

report = classification_report(
    y_test,
    y_pred
)

print(report)


cm = confusion_matrix(
    y_test,
    y_pred
)

print("\nConfusion Matrix")

print(cm)

y_prob = model.predict_proba(
    X_test
)[:,1]

fpr, tpr, threshold = roc_curve(
    y_test,
    y_prob
)

roc_auc = auc(
    fpr,
    tpr
)

plt.figure(figsize=(8,6))

plt.plot(
    fpr,
    tpr,
    label=f"AUC = {roc_auc:.4f}"
)

plt.plot(
    [0,1],
    [0,1],
    "r--"
)

plt.xlabel(
    "False Positive Rate"
)

plt.ylabel(
    "True Positive Rate"
)

plt.title(
    "ROC Curve"
)

plt.legend()

plt.savefig(
    "roc_curve.png"
)

plt.show()




plt.figure(figsize=(6,5))

sns.heatmap(
    cm,
    annot=True,
    fmt='d'
)

plt.title("Confusion Matrix")

plt.xlabel("Predicted")

plt.ylabel("Actual")

plt.savefig("confusion_matrix.png")

plt.show()


importance = model.feature_importances_

feature_importance = pd.DataFrame({
    'Feature':X.columns,
    'Importance':importance
})

feature_importance = feature_importance.sort_values(
    by='Importance',
    ascending=False
)

print("\nTop 10 Important Features")

print(feature_importance.head(10))

feature_importance.to_csv(
    "feature_importance.csv",
    index=False
)

plt.figure(figsize=(10,6))

sns.barplot(
    x='Importance',
    y='Feature',
    data=feature_importance.head(10)
)

plt.title("Top 10 Important Features")

plt.show()


with open(
    "model_results.txt",
    "w"
) as file:

    file.write(
        "Credit Card Fraud Detection Results\n\n"
    )

    file.write(
        f"Accuracy : {accuracy}\n\n"
    )

    file.write(report)

print("\nResults Saved Successfully")

print("\nChecking Sample Transaction...")

sample = X_test.iloc[[0]]

prediction = model.predict(sample)

if prediction[0] == 1:
    print("Fraud Detected")
else:
    print("Genuine Transaction")

print("\nProject Completed Successfully")