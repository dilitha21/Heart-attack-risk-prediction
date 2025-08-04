from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import joblib
import os
import traceback
from datetime import datetime

# Initialize Flask app
app = Flask(__name__)

# Enable CORS with explicit configuration
CORS(app, resources={
    r"/*": {
        "origins": ["*"],
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type"]
    }
})

# Add OPTIONS handler for preflight requests
@app.before_request  
def handle_preflight():
    if request.method == "OPTIONS":
        response = jsonify({})
        response.headers.add("Access-Control-Allow-Origin", "*")
        response.headers.add('Access-Control-Allow-Headers', "*")
        response.headers.add('Access-Control-Allow-Methods', "*")
        return response

# Global variables to store loaded models
model = None
scaler = None
label_encoders = None
feature_columns = None

def load_model_components():
    """Load saved model components at startup"""
    global model, scaler, label_encoders
    
    print("Loading model components...")
    
    try:
        # Load the trained Random Forest model
        model_path = 'model/rf_model.pkl'
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        model = joblib.load(model_path)
        print(f" Model loaded from {model_path}")
        
        # Load the StandardScaler
        scaler_path = 'model/scaler.pkl'
        if not os.path.exists(scaler_path):
            raise FileNotFoundError(f"Scaler file not found: {scaler_path}")
        scaler = joblib.load(scaler_path)
        print(f" Scaler loaded from {scaler_path}")
        
        # Load the label encoders
        encoders_path = 'model/label_encoders.pkl'
        if not os.path.exists(encoders_path):
            raise FileNotFoundError(f"Label encoders file not found: {encoders_path}")
        label_encoders = joblib.load(encoders_path)
        print(f" Label encoders loaded from {encoders_path}")
        
        # Get feature names from the model (if available)
        if hasattr(model, 'feature_names_in_'):
            global feature_columns
            feature_columns = model.feature_names_in_.tolist()
            print(f" Feature columns: {feature_columns}")
        
        print("All model components loaded successfully!")
        return True
        
    except Exception as e:
        print(f" Error loading model components: {str(e)}")
        print("Make sure to run train_model.py first to generate the required files.")
        return False

def validate_input_data(data):
    """Validate that required fields are present in input data"""
    required_fields = [
        'Age', 'Gender', 'Cholesterol', 'Blood Pressure', 'Heart Rate',
        'Smoking', 'Obesity', 'Diabetes', 'Previous Heart Problems', 
        'Medication Use', 'Exercise Hours Per Week', 'Stress Level'
    ]
    
    missing_fields = []
    for field in required_fields:
        if field not in data:
            missing_fields.append(field)
    
    if missing_fields:
        return False, f"Missing required fields: {missing_fields}"
    
    return True, "All required fields present"

def map_feature_names(data):
    """Map frontend field names to training field names"""
    # Create a mapping from frontend names to training names
    name_mapping = {
        'Cholesterol': 'Cholesterol Level',
        'Exercise Hours Per Week': 'Exercise Hours Per Week',
        'Stress Level': 'Stress Level'
    }
    
    # Create new data dict with mapped names
    mapped_data = {}
    for key, value in data.items():
        # Use mapped name if exists, otherwise use original name
        mapped_key = name_mapping.get(key, key)
        mapped_data[mapped_key] = value
    
    return mapped_data

def preprocess_input(data):
    """Preprocess input data same way as training data"""
    try:
        # First, map feature names to match training data
        mapped_data = map_feature_names(data)
        
        # Create DataFrame from mapped input data
        df = pd.DataFrame([mapped_data])
        
        # Print current columns for debugging
        print(f"DataFrame columns after mapping: {list(df.columns)}")
        
        # Define categorical columns that need encoding
        categorical_columns = ['Gender', 'Smoking', 'Obesity', 'Diabetes', 
                             'Previous Heart Problems', 'Medication Use']
        
        # Encode categorical variables using saved encoders
        for col in categorical_columns:
            if col in df.columns and col in label_encoders:
                encoder = label_encoders[col]
                try:
                    # Handle new/unseen values by using the most frequent class
                    if str(df[col].iloc[0]) not in encoder.classes_:
                        print(f"Warning: Unseen value '{df[col].iloc[0]}' for {col}, using most frequent class")
                        # Use the first class as default (you might want to use mode from training)
                        df[col] = encoder.classes_[0]
                    
                    df[col] = encoder.transform(df[col].astype(str))
                except Exception as e:
                    print(f"Error encoding {col}: {str(e)}")
                    # Use default value (0) if encoding fails
                    df[col] = 0
        
        # Ensure all numeric columns are properly typed
        numeric_columns = ['Age', 'Cholesterol Level', 'Blood Pressure', 'Heart Rate', 
                          'Exercise Hours Per Week', 'Stress Level']
        
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                # Fill any NaN values with 0
                df[col] = df[col].fillna(0)
        
        # If we have feature_columns from the model, ensure correct order
        if feature_columns:
            print(f"Expected feature columns: {feature_columns}")
            print(f"Current DataFrame columns: {list(df.columns)}")
            
            # Reorder columns to match training order
            try:
                df = df[feature_columns]
            except KeyError as e:
                missing_cols = [col for col in feature_columns if col not in df.columns]
                print(f"Missing columns: {missing_cols}")
                # Add missing columns with default values
                for col in missing_cols:
                    df[col] = 0
                df = df[feature_columns]
        
        # Scale the features using the saved scaler
        df_scaled = scaler.transform(df)
        
        return df_scaled, None
        
    except Exception as e:
        print(f"Full preprocessing error: {str(e)}")
        print(f"Input data: {data}")
        return None, f"Preprocessing error: {str(e)}"

@app.route('/', methods=['GET'])
def home():
    """Health check endpoint"""
    return jsonify({
        "message": "Heart Attack Risk Prediction API",
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "model_loaded": model is not None
    })

@app.route('/model-info', methods=['GET'])
def model_info():
    """Get information about the loaded model"""
    if model is None:
        return jsonify({"error": "Model not loaded"}), 500
    
    info = {
        "model_type": str(type(model).__name__),
        "feature_count": len(feature_columns) if feature_columns else "Unknown",
        "categorical_encoders": list(label_encoders.keys()) if label_encoders else [],
        "scaler_loaded": scaler is not None
    }
    
    if hasattr(model, 'n_estimators'):
        info["n_estimators"] = model.n_estimators
    if hasattr(model, 'max_depth'):
        info["max_depth"] = model.max_depth
        
    return jsonify(info)

@app.route('/predict', methods=['POST'])
def predict():
    """Main prediction endpoint"""
    try:
        # Check if models are loaded
        if model is None or scaler is None or label_encoders is None:
            return jsonify({
                "error": "Model components not loaded. Please restart the server."
            }), 500
        
        # Get JSON data from request
        if not request.is_json:
            return jsonify({"error": "Request must be JSON"}), 400
        
        data = request.get_json()
        
        if not data:
            return jsonify({"error": "No data provided"}), 400
        
        # Validate input data
        is_valid, validation_message = validate_input_data(data)
        if not is_valid:
            return jsonify({"error": validation_message}), 400
        
        # Preprocess the input data
        processed_data, error_message = preprocess_input(data)
        
        if processed_data is None:
            return jsonify({"error": error_message}), 400
        
        # Make prediction
        prediction = model.predict(processed_data)
        prediction_proba = model.predict_proba(processed_data)
        
        # Get the predicted risk (0 or 1)
        risk = int(prediction[0])
        
        # Get probability scores
        risk_probabilities = {
            "low_risk_probability": float(prediction_proba[0][0]),
            "high_risk_probability": float(prediction_proba[0][1])
        }
        
        # Prepare response
        response = {
            "risk": risk,
            "risk_label": "High Risk" if risk == 1 else "Low Risk",
            "confidence": float(max(prediction_proba[0])),
            "probabilities": risk_probabilities,
            "timestamp": datetime.now().isoformat()
        }
        
        print(f"Prediction made: {response}")
        
        return jsonify(response)
        
    except Exception as e:
        error_message = f"Prediction error: {str(e)}"
        print(f" {error_message}")
        print(traceback.format_exc())
        
        return jsonify({
            "error": error_message,
            "timestamp": datetime.now().isoformat()
        }), 500

@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return jsonify({"error": "Endpoint not found"}), 404

@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    return jsonify({"error": "Internal server error"}), 500

if __name__ == '__main__':
    print("=== Heart Attack Risk Prediction API ===\n")
    
    # Load model components at startup
    if load_model_components():
        print(f"\nStarting Flask server on http://localhost:5000")
        print("Available endpoints:")
        print("  GET  /          - Health check")
        print("  GET  /model-info - Model information")
        print("  POST /predict   - Make predictions")
        print("\nExample prediction request:")
        print("""
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "Age": 45,
    "Gender": "Male",
    "Cholesterol": 240,
    "Blood Pressure": "140/90",
    "Heart Rate": 80,
    "Smoking": "Yes",
    "Obesity": "No",
    "Diabetes": "No",
    "Previous Heart Problems": "No",
    "Medication Use": "Yes",
    "Exercise Hours Per Week": 2,
    "Stress Level": 7
  }'
        """)
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        print("\nFailed to load model components. Cannot start server.")
        print("Please run 'python train_model.py' first to train and save the model.")