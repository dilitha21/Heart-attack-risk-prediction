import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

# Load the dataset
dataset_path = 'heart-attack-risk-prediction-dataset.csv'
df = pd.read_csv(dataset_path)

# Show basic info
print("Dataset shape:", df.shape)
print("Columns:", list(df.columns))
print(df.head())
print(df.isnull().sum())

# Import necessary libraries for ML
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler, LabelEncoder

# Check for missing values

# Fill missing values for numeric columns
for col in df.select_dtypes(include=[np.number]).columns:
    df[col].fillna(df[col].median(), inplace=True)

# Fill missing values for object columns
for col in df.select_dtypes(include=['object']).columns:
    df[col].fillna(df[col].mode()[0], inplace=True)

# Categorical columns (use only those present in the dataset)
categorical_cols = [col for col in ['Gender', 'Smoking', 'Obesity', 'Diabetes', 'Previous Heart Problems', 'Medication Use'] if col in df.columns]
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le

# Define feature columns and target variable

# Features and target
target_col = 'Heart Attack Risk'
feature_cols = [col for col in df.columns if col != target_col]
X = df[feature_cols]
y = df[target_col]

# Normalize numerical features

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split dataset into training (80%) and testing (20%)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

# Print dataset shape
print(f"Training Data Shape: {X_train.shape}, Testing Data Shape: {X_test.shape}")

# Import ML libraries

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report

# Train Logistic Regression

# Train Logistic Regression
log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X_train, y_train)
y_pred_log = log_reg.predict(X_test)

# Train Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

# Train XGBoost
xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
xgb.fit(X_train, y_train)
y_pred_xgb = xgb.predict(X_test)

# Evaluate Models

print("Logistic Regression Accuracy:", accuracy_score(y_test, y_pred_log))
print("Random Forest Accuracy:", accuracy_score(y_test, y_pred_rf))
print("XGBoost Accuracy:", accuracy_score(y_test, y_pred_xgb))

print("\nLogistic Regression Report:\n", classification_report(y_test, y_pred_log))
print("\nRandom Forest Report:\n", classification_report(y_test, y_pred_rf))
print("\nXGBoost Report:\n", classification_report(y_test, y_pred_xgb))

# Random Forest Hyperparameter Tuning

## Hyperparameter tuning code removed for simplicity and compatibility

# Model Evaluation

# Save all three models for comparison
if not os.path.exists('model'):
    os.makedirs('model')
    print("Created 'model' directory")

joblib.dump(log_reg, 'model/log_reg_model.pkl')
print("Logistic Regression model saved to model/log_reg_model.pkl")

joblib.dump(rf, 'model/rf_model.pkl')
print("Random Forest model saved to model/rf_model.pkl")

joblib.dump(xgb, 'model/xgb_model.pkl')
print("XGBoost model saved to model/xgb_model.pkl")

joblib.dump(scaler, 'model/scaler.pkl')
print("Scaler saved to model/scaler.pkl")

joblib.dump(label_encoders, 'model/label_encoders.pkl')
print("Label encoders saved to model/label_encoders.pkl")

feature_names = X.columns.tolist()
joblib.dump(feature_names, 'model/feature_names.pkl')
print("Feature names saved to model/feature_names.pkl")

print("\nML Training Complete!")
print("Results Summary:")
print(f"  Logistic Regression Accuracy: {accuracy_score(y_test, y_pred_log):.4f}")
print(f"  Random Forest Accuracy: {accuracy_score(y_test, y_pred_rf):.4f}")
print(f"  XGBoost Accuracy: {accuracy_score(y_test, y_pred_xgb):.4f}")
print("\nReady for API deployment!")
print("  Run: python api.py")

# ======== NEW SECTION: SAVE MODELS FOR API ========

print("\n" + "="*50)
print("SAVING RANDOM FOREST MODEL FOR API DEPLOYMENT")
print("="*50)

# Create model directory
if not os.path.exists('model'):
    os.makedirs('model')
    print("Created 'model' directory")

# Save only the Random Forest model for API compatibility
joblib.dump(rf, 'model/rf_model.pkl')
print(" Random Forest model saved to model/rf_model.pkl")

# Save the scaler
joblib.dump(scaler, 'model/scaler.pkl')
print(" Scaler saved to model/scaler.pkl")

# Save label encoders
joblib.dump(label_encoders, 'model/label_encoders.pkl')
print(" Label encoders saved to model/label_encoders.pkl")

# Save feature names for API reference
feature_names = X.columns.tolist()
joblib.dump(feature_names, 'model/feature_names.pkl')
print(" Feature names saved to model/feature_names.pkl")

print(f"\nML Training Complete!")
print("Results Summary:")
print(f"  Logistic Regression Accuracy: {accuracy_score(y_test, y_pred_log):.4f}")
print(f"  Random Forest Accuracy: {accuracy_score(y_test, y_pred_rf):.4f}")
print(f"  XGBoost Accuracy: {accuracy_score(y_test, y_pred_xgb):.4f}")
print("\nReady for API deployment!")
print("  Run: python api.py")