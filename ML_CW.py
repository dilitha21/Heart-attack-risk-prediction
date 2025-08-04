import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
dataset_path = 'heart-attack-risk-prediction-dataset.csv'
df = pd.read_csv(dataset_path)

# Display basic information about the dataset
print("Dataset Info:")
df.info()

# Check for missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Display first few rows of the dataset
print("\nFirst 5 Rows:")
print(df.head())

# Summary statistics of numerical columns
print("\nSummary Statistics:")
print(df.describe())

# Visualizations
plt.figure(figsize=(10, 6))
sns.histplot(df['Age'], bins=20, kde=True, color='blue')
plt.title('Age Distribution')
plt.xlabel('Age')
plt.ylabel('Count')
plt.show()

plt.figure(figsize=(8, 5))
sns.countplot(x='Gender', data=df, palette='coolwarm')
plt.title('Gender Distribution')
plt.xlabel('Gender (0 = Female, 1 = Male)')
plt.ylabel('Count')
plt.show()

plt.figure(figsize=(8, 5))
sns.countplot(x='Heart Attack Risk', data=df, palette='Set2')
plt.title('Heart Attack Risk Distribution')
plt.xlabel('Heart Attack Risk (0 = Low, 1 = High)')
plt.ylabel('Count')
plt.show()

# Correlation heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Feature Correlation Heatmap')
plt.show()


# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Load the dataset
file_path = "heart-attack-risk-prediction-dataset.csv"
df = pd.read_csv(file_path)

# Check for missing values
print("Missing Values:\n", df.isnull().sum())

# Fill missing values (if any)
df.fillna(df.median(), inplace=True)  # Replace numerical NaNs with median

# Encoding categorical variables (if any exist)
categorical_cols = ['Gender', 'Smoking', 'Obesity', 'Diabetes', 'Previous Heart Problems', 'Medication Use']
label_encoders = {}

for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le  # Store encoder for later use

# Define feature columns and target variable
X = df.drop(columns=["Heart Attack Risk"])  # Features
y = df["Heart Attack Risk"]  # Target

# Normalize numerical features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split dataset into training (80%) and testing (20%)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

# Print dataset shape
print(f"Training Data Shape: {X_train.shape}, Testing Data Shape: {X_test.shape}")


# Import necessary libraries
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV

# Train Logistic Regression
log_reg = LogisticRegression()
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

# Print Classification Reports
print("\nLogistic Regression Report:\n", classification_report(y_test, y_pred_log))
print("\nRandom Forest Report:\n", classification_report(y_test, y_pred_rf))
print("\nXGBoost Report:\n", classification_report(y_test, y_pred_xgb))


#Random Forest Hyperparameter Tuning

# Define hyperparameter grid
param_grid_rf = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10]
}

# Perform Grid Search
grid_rf = GridSearchCV(RandomForestClassifier(random_state=42), param_grid_rf, cv=3, scoring='accuracy')
grid_rf.fit(X_train, y_train)

# Best parameters and accuracy
print("Best Random Forest Parameters:", grid_rf.best_params_)
print("Best Random Forest Accuracy:", grid_rf.best_score_)


#XGBoost Hyperparameter Tuning
# Define hyperparameter grid
param_grid_xgb = {
    'learning_rate': [0.01, 0.1, 0.2],
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 5, 7]
}

# Perform Grid Search
grid_xgb = GridSearchCV(XGBClassifier(use_label_encoder=False, eval_metric='logloss'), param_grid_xgb, cv=3, scoring='accuracy')
grid_xgb.fit(X_train, y_train)

# Best parameters and accuracy
print("Best XGBoost Parameters:", grid_xgb.best_params_)
print("Best XGBoost Accuracy:", grid_xgb.best_score_)

#Model Evaluation

# Import necessary libraries
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve
import matplotlib.pyplot as plt

# Function to evaluate models
def evaluate_model(model_name, y_true, y_pred, y_probs):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    roc_auc = roc_auc_score(y_true, y_probs)

    print(f"\n{model_name} Evaluation Metrics:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-score: {f1:.4f}")
    print(f"ROC-AUC: {roc_auc:.4f}")

    return accuracy, precision, recall, f1, roc_auc

# Get probability scores for ROC-AUC calculation
y_probs_log = log_reg.predict_proba(X_test)[:,1]
y_probs_rf = rf.predict_proba(X_test)[:,1]
y_probs_xgb = xgb.predict_proba(X_test)[:,1]

# Evaluate each model
metrics_log = evaluate_model("Logistic Regression", y_test, y_pred_log, y_probs_log)
metrics_rf = evaluate_model("Random Forest", y_test, y_pred_rf, y_probs_rf)
metrics_xgb = evaluate_model("XGBoost", y_test, y_pred_xgb, y_probs_xgb)

# Compute ROC curves
fpr_log, tpr_log, _ = roc_curve(y_test, y_probs_log)
fpr_rf, tpr_rf, _ = roc_curve(y_test, y_probs_rf)
fpr_xgb, tpr_xgb, _ = roc_curve(y_test, y_probs_xgb)

# Plot ROC Curves
plt.figure(figsize=(8,6))
plt.plot(fpr_log, tpr_log, label="Logistic Regression (AUC = {:.4f})".format(metrics_log[4]))
plt.plot(fpr_rf, tpr_rf, label="Random Forest (AUC = {:.4f})".format(metrics_rf[4]))
plt.plot(fpr_xgb, tpr_xgb, label="XGBoost (AUC = {:.4f})".format(metrics_xgb[4]))

# Add plot details
plt.plot([0,1], [0,1], 'k--')  # Diagonal line for random performance
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve Comparison")
plt.legend()
plt.show()
