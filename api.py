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
CORS(app, resources={
    r"/*": {
        "origins": ["*"],
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type"]
    }
})

# Global variables
model = None
scaler = None
label_encoders = None
feature_names = None
model_info = None

def load_model_components():
    """Load saved model components from ML_CW.py"""
    global model, scaler, label_encoders, feature_names, model_info
    
    print("Loading model components from ML_CW.py output...")
    
    try:
        # Load the Random Forest model (as saved by your ML_CW.py)
        model_path = 'model/rf_model.pkl'
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        model = joblib.load(model_path)
        print(f" Random Forest model loaded from {model_path}")
        
        # Load the scaler
        scaler_path = 'model/scaler.pkl'
        if not os.path.exists(scaler_path):
            raise FileNotFoundError(f"Scaler file not found: {scaler_path}")
        scaler = joblib.load(scaler_path)
        print(f" Scaler loaded from {scaler_path}")
        
        # Load label encoders
        encoders_path = 'model/label_encoders.pkl'
        if not os.path.exists(encoders_path):
            raise FileNotFoundError(f"Encoders file not found: {encoders_path}")
        label_encoders = joblib.load(encoders_path)
        print(f" Label encoders loaded from {encoders_path}")
        
        # Load feature names
        feature_names_path = 'model/feature_names.pkl'
        if os.path.exists(feature_names_path):
            feature_names = joblib.load(feature_names_path)
            print(f" Feature names loaded: {feature_names}")
        
        # Create model info since it's not saved separately
        model_info = {
            'best_model_name': 'Random Forest',
            'best_accuracy': 'Unknown (check ML_CW.py output)',
            'feature_names': feature_names,
            'categorical_columns': list(label_encoders.keys()) if label_encoders else [],
            'target_column': 'Heart Attack Risk',
            'training_date': 'Unknown'
        }
        print(f" Model info created for Random Forest")
        
        print("All model components loaded successfully!")
        return True
        
    except Exception as e:
        print(f" Error loading model components: {str(e)}")
        print("Make sure to run ML_CW.py first to generate the required model files.")
        return False

@app.route('/', methods=['GET'])
def home():
    """Health check endpoint"""
    return jsonify({
        "message": "Heart Attack Risk Prediction API (Random Forest)",
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "model_loaded": model is not None,
        "model_type": "Random Forest"
    })

@app.route('/model-info', methods=['GET'])
def get_model_info():
    """Get detailed model information"""
    if not model_info:
        return jsonify({"error": "Model info not available"}), 500
    
    return jsonify({
        "model_name": model_info.get('best_model_name', 'Unknown'),
        "accuracy": model_info.get('best_accuracy', 0),
        "feature_count": len(feature_names) if feature_names else 0,
        "features": feature_names,
        "categorical_encoders": list(label_encoders.keys()) if label_encoders else [],
        "training_date": model_info.get('training_date', 'Unknown'),
        "target": model_info.get('target_column', 'Heart Attack Risk')
    })

def preprocess_input(data):
    """Preprocess input data same way as ML_CW.py"""
    try:
        # Create DataFrame from input data
        df = pd.DataFrame([data])
        print(f"Input data: {data}")
        
        # Get expected feature names
        if feature_names:
            print(f"Expected features: {feature_names}")
        
        # Encode categorical variables using saved encoders
        if label_encoders:
            for col, encoder in label_encoders.items():
                if col in df.columns:
                    try:
                        # Handle unseen values
                        if str(df[col].iloc[0]) not in encoder.classes_:
                            print(f"Warning: Unseen value '{df[col].iloc[0]}' for {col}")
                            df[col] = encoder.classes_[0]  # Use first class as default
                        
                        df[col] = encoder.transform(df[col].astype(str))
                        print(f"Encoded {col}: {df[col].iloc[0]}")
                    except Exception as e:
                        print(f"Error encoding {col}: {e}")
                        df[col] = 0
        
        # Ensure all numeric columns are properly typed
        for col in df.columns:
            if col not in (label_encoders.keys() if label_encoders else []):
                df[col] = pd.to_numeric(df[col], errors='coerce')
                df[col] = df[col].fillna(0)
        
        # Reorder columns to match training order if we have feature names
        if feature_names:
            # Add missing columns with default values
            for col in feature_names:
                if col not in df.columns:
                    df[col] = 0
                    print(f"Added missing column {col} with value 0")
            
            # Reorder to match training
            df = df[feature_names]
        
        print(f"Final DataFrame shape: {df.shape}")
        print(f"Final DataFrame columns: {list(df.columns)}")
        
        # Scale features
        df_scaled = scaler.transform(df)
        print("Features scaled successfully")
        
        return df_scaled, None
        
    except Exception as e:
        error_msg = f"Preprocessing error: {str(e)}"
        print(error_msg)
        print(traceback.format_exc())
        return None, error_msg

@app.route('/predict', methods=['POST'])
def predict():
    """Make heart attack risk prediction"""
    try:
        # Check if components are loaded
        if not all([model, scaler, label_encoders]):
            return jsonify({
                "error": "Model components not loaded. Run ML_CW.py first."
            }), 500
        
        # Get JSON data
        if not request.is_json:
            return jsonify({"error": "Request must be JSON"}), 400
        
        data = request.get_json()
        if not data:
            return jsonify({"error": "No data provided"}), 400
        
        print(f"Received prediction request: {data}")
        
        # Preprocess input
        processed_data, error_msg = preprocess_input(data)
        if processed_data is None:
            return jsonify({"error": error_msg}), 400
        
        # Make prediction
        prediction = model.predict(processed_data)
        prediction_proba = model.predict_proba(processed_data)
        
        # Format response
        risk = int(prediction[0])
        confidence = float(max(prediction_proba[0]))
        
        response = {
            "risk": risk,
            "risk_label": "High Risk" if risk == 1 else "Low Risk",
            "confidence": confidence,
            "probabilities": {
                "low_risk": float(prediction_proba[0][0]),
                "high_risk": float(prediction_proba[0][1])
            },
            "model_used": model_info['best_model_name'] if model_info else "Unknown",
            "timestamp": datetime.now().isoformat()
        }
        
        print(f"Prediction result: {response}")
        return jsonify(response)
        
    except Exception as e:
        error_msg = f"Prediction failed: {str(e)}"
        print(error_msg)
        print(traceback.format_exc())
        return jsonify({"error": error_msg}), 500

@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Endpoint not found"}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({"error": "Internal server error"}), 500

if __name__ == '__main__':
    print("=== Heart Attack Risk Prediction API (ML_CW Version) ===\n")
    
    if load_model_components():
        print(f"\n Starting Flask server on http://localhost:5000")
        print("Available endpoints:")
        print("  GET  /          - Health check")
        print("  GET  /model-info - Model information")
        print("  POST /predict   - Make predictions")
        
        if model_info:
            print(f"\nUsing: Random Forest Model")
        
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        print("\n Failed to load model components.")
        print("Please run ML_CW.py first to train and save models.")