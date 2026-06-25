# Credit Card Fraud Detection using Machine Learning

## overview

This project aims to detect fraudulent credit card transactions using Machine Learning techniques. The model is trained on a real-world credit card transaction dataset and classifies transactions as either genuine or fraudulent.

The project was developed as part of the CodeSoft Machine Learning Internship.

## Project Objectives

* Analyze credit card transaction data.
* Identify fraudulent transactions.
* Build a Machine Learning model for classification.
* Evaluate model performance using multiple metrics.
* Visualize important patterns and insights.

## Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-Learn

## Installation


git clone https://github.com/yourusername/Credit-Card-Fraud-Detection.git

cd Credit-Card-Fraud-Detection

pip install -r requirements.txt

## 5. How to Run

## Run Project

python fraud_detection.py


## Dataset Information

The dataset contains:

* 284,807 transactions
* 31 columns
* 492 fraudulent transactions
* 284,315 genuine transactions

### Target Variable

* Class = 0 → Genuine Transaction
* Class = 1 → Fraudulent Transaction

## Project Workflow

### 1. Data Collection

Loaded the credit card transaction dataset using Pandas.

### 2. Data Exploration

* Dataset Shape
* Dataset Information
* Missing Values Check
* Class Distribution Analysis

### 3. Data Visualization

* Fraud vs Genuine Transaction Count Plot
* Correlation Heatmap
* Confusion Matrix
* ROC Curve
* Feature Importance Graph

### 4. Data Preprocessing

* Feature Scaling using StandardScaler
* Train-Test Split (80:20)

### 5. Model Building

Random Forest Classifier was used for fraud detection.

### 6. Model Evaluation

Performance metrics:

* Accuracy Score
* Precision Score
* Recall Score
* F1 Score
* Classification Report
* Confusion Matrix
* ROC-AUC Score

### 7. Prediction

The trained model predicts whether a transaction is:

* Genuine Transaction
* Fraudulent Transaction
  
## Results

Example Performance:

* Accuracy: 99.95%
* Precision: 94%
* Recall: 82%
* F1 Score: 87%

The model successfully identifies fraudulent transactions with high accuracy.

## Generated Files

After running the project, the following files are generated:

* class_distribution.png
* correlation_heatmap.png
* confusion_matrix.png
* roc_curve.png
* feature_importance.csv
* model_results.txt

##  Output

Dataset Loaded Successfully

Accuracy : 0.9995

Precision : 0.94
Recall : 0.82
F1 Score : 0.87

Checking Sample Transaction...
Genuine Transaction

Project Completed Successfully

## Project Screenshots


## Dataset Overview

<img width="578" height="435" alt="image" src="https://github.com/user-attachments/assets/bf77cf98-2f1b-4143-bb5e-668facd92b40" />

<img width="263" height="446" alt="image" src="https://github.com/user-attachments/assets/530f94c1-4f73-48a4-a881-532a0616762b" />

<img width="273" height="351" alt="image" src="https://github.com/user-attachments/assets/5bf2760e-1089-4b07-a51f-f75cb1cc30db" />

<img width="359" height="333" alt="image" src="https://github.com/user-attachments/assets/09898301-92c3-4eaf-b12f-ae6fb8dbeb30" />

## Fraud vs Genuine Transactions


<img width="512" height="389" alt="image" src="https://github.com/user-attachments/assets/07ef0bec-8a98-4a4d-a2da-2dd45502f678" />

## Correlation Heatmap


<img width="797" height="446" alt="image" src="https://github.com/user-attachments/assets/cc442e03-5509-4656-858d-692c54f47712" />

<img width="847" height="434" alt="image" src="https://github.com/user-attachments/assets/9424e34e-a8dd-49d3-94e9-21a3d2e5081a" />


## ROC Curve


<img width="781" height="443" alt="image" src="https://github.com/user-attachments/assets/131a2197-8436-4f5a-8afd-8ab2cb5667c6" />

##  Confusion Matrix


<img width="845" height="439" alt="image" src="https://github.com/user-attachments/assets/b1e28e5f-870f-4236-b325-52de018471db" />


## Applications

* Banking Systems
* Online Payment Platforms
* Financial Security Systems
* Fraud Monitoring Solutions
* Risk Management Systems

## Future Improvements

* Implement Deep Learning Models
* Real-Time Fraud Detection
* Deploy as a Web Application
* Integrate with Banking APIs
* Improve Fraud Recall Rate

## Author

Pratiksha Tomar

Machine Learning Intern – CodeSoft
