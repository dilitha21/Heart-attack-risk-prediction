import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import os

def create_model_directory():
    """Create model directory if it doesn't exist"""
    if not os.path.exists('model'):
        os.makedirs('model')
        print("Created 'model' directory")

def load_and_explore_data(file_path):
    """Load dataset and display basic information"""
    print("Loading dataset...")
    df = pd.read_csv(file_path)
    
    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print("\nFirst 5 rows:")
    print(df.head())
    
    print("\nMissing values:")
    print(df.isnull().sum())
    
    print("\nData types:")
    print(df.dtypes)
    
    # Standardize column names to match frontend expectations
    column_mapping = {
        'Cholesterol Level': 'Cholesterol Level',  # Keep as is if already correct
        'Cholesterol': 'Cholesterol Level',        # Rename if needed
    }
    
    # Apply column renaming if needed
    df = df.rename(columns=column_mapping)
    print(f"\nFinal columns after standardization: {list(df.columns)}")
    
    return df

def preprocess_data(df):
    """Handle missing values and encode categorical variables"""
    print("\n=== Data Preprocessing ===")
    
    # Make a copy to avoid modifying original data
    df_processed = df.copy()
    
    # Fill missing values with median for numerical columns
    numerical_cols = df_processed.select_dtypes(include=[np.number]).columns
    for col in numerical_cols:
        if df_processed[col].isnull().sum() > 0:
            median_val = df_processed[col].median()
            df_processed[col].fillna(median_val, inplace=True)
            print(f"Filled {col} missing values with median: {median_val}")
    
    # Fill missing values with mode for categorical columns
    categorical_cols = df_processed.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        if df_processed[col].isnull().sum() > 0:
            mode_val = df_processed[col].mode()[0]
            df_processed[col].fillna(mode_val, inplace=True)
            print(f"Filled {col} missing values with mode: {mode_val}")
    
    # Identify categorical columns that need encoding
    # Based on your original code, these are the expected categorical columns
    categorical_columns = ['Gender', 'Smoking', 'Obesity', 'Diabetes', 
                          'Previous Heart Problems', 'Medication Use']
    
    # Filter only existing columns
    existing_categorical = [col for col in categorical_columns if col in df_processed.columns]
    
    if not existing_categorical:
        # If no predefined categorical columns found, identify them automatically
        existing_categorical = df_processed.select_dtypes(include=['object']).columns.tolist()
        # Remove target column if it's in categorical
        if 'Heart Attack Risk' in existing_categorical:
            existing_categorical.remove('Heart Attack Risk')
    
    print(f"Categorical columns to encode: {existing_categorical}")
    
    # Encode categorical variables
    label_encoders = {}
    for col in existing_categorical:
        le = LabelEncoder()
        df_processed[col] = le.fit_transform(df_processed[col].astype(str))
        label_encoders[col] = le
        print(f"Encoded {col}: {dict(zip(le.classes_, le.transform(le.classes_)))}")
    
    print(f"Total missing values after preprocessing: {df_processed.isnull().sum().sum()}")
    
    return df_processed, label_encoders

def prepare_features_target(df):
    """Separate features and target variable"""
    print("\n=== Feature Engineering ===")
    
    # Define target column
    target_col = "Heart Attack Risk"
    
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataset!")
    
    # Separate features and target
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    print(f"Features shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    print(f"Feature columns: {list(X.columns)}")
    print(f"Target distribution:\n{y.value_counts()}")
    
    return X, y

def scale_features(X_train, X_test):
    """Scale features using StandardScaler"""
    print("\n=== Feature Scaling ===")
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print("Features scaled using StandardScaler")
    print(f"Training data shape after scaling: {X_train_scaled.shape}")
    print(f"Test data shape after scaling: {X_test_scaled.shape}")
    
    return X_train_scaled, X_test_scaled, scaler

def train_random_forest(X_train, y_train, X_test, y_test):
    """Train Random Forest with hyperparameter tuning"""
    print("\n=== Model Training ===")
    
    # Define parameter grid for hyperparameter tuning
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    
    # Initialize Random Forest
    rf = RandomForestClassifier(random_state=42)
    
    # Perform Grid Search with cross-validation
    print("Performing hyperparameter tuning...")
    grid_search = GridSearchCV(
        estimator=rf,
        param_grid=param_grid,
        cv=5,
        scoring='accuracy',
        n_jobs=-1,
        verbose=1
    )
    
    # Fit the grid search
    grid_search.fit(X_train, y_train)
    
    # Get the best model
    best_model = grid_search.best_estimator_
    
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best cross-validation score: {grid_search.best_score_:.4f}")
    
    # Make predictions
    y_pred = best_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"Test accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    return best_model

def save_model_components(model, scaler, label_encoders):
    """Save trained model and preprocessing components"""
    print("\n=== Saving Model Components ===")
    
    # Save the trained model
    joblib.dump(model, 'model/rf_model.pkl')
    print("✓ Random Forest model saved to model/rf_model.pkl")
    
    # Save the scaler
    joblib.dump(scaler, 'model/scaler.pkl')
    print("✓ StandardScaler saved to model/scaler.pkl")
    
    # Save the label encoders
    joblib.dump(label_encoders, 'model/label_encoders.pkl')
    print("✓ Label encoders saved to model/label_encoders.pkl")
    
    print("\nAll model components saved successfully!")

def main():
    """Main training pipeline"""
    print("=== Heart Attack Risk Prediction - Model Training ===\n")
    
    # Create model directory
    create_model_directory()
    
    # Load and explore data
    dataset_path = 'heart-attack-risk-prediction-dataset.csv'
    
    try:
        df = load_and_explore_data(dataset_path)
    except FileNotFoundError:
        print(f"Error: Dataset file '{dataset_path}' not found!")
        print("Please ensure the dataset is in the same directory as this script.")
        return
    
    # Preprocess data
    df_processed, label_encoders = preprocess_data(df)
    
    # Prepare features and target
    X, y = prepare_features_target(df_processed)
    
    # Split the data
    print("\n=== Data Splitting ===")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    # Scale features
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)
    
    # Train model
    model = train_random_forest(X_train_scaled, y_train, X_test_scaled, y_test)
    
    # Save all components
    save_model_components(model, scaler, label_encoders)
    
    print("\n🎉 Model training completed successfully!")
    print("\nYou can now use the trained model in your Flask API.")

if __name__ == "__main__":
    main()