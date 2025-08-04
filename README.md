# Heart Attack Risk Prediction

A web application that predicts the risk of heart attack based on user health and lifestyle information using a machine learning model.

## Features
- Interactive web frontend for user input
- Flask API backend for prediction
- Machine learning model (Random Forest) trained on health data
- Consistent field names across frontend, backend, and model

## Project Structure

```
Heart-Attack-Risk-Prediction/
├── api.py                  # Flask API backend
├── train_model.py          # Model training script
├── requirements.txt        # Python dependencies
├── heart-attack-risk-prediction-dataset.csv  # Dataset
├── model/                  # Saved model, scaler, encoders
│   ├── rf_model.pkl
│   ├── scaler.pkl
│   └── label_encoders.pkl
├── frontend/
│   ├── index.html          # Main web page
│   ├── css/
│   │   └── style.css
│   ├── js/
│   │   └── app.js
│   └── assets/
│       ├── fonts/
│       └── images/
└── README.md
```

## Setup Instructions

1. **Clone the repository:**
   ```sh
   git clone https://github.com/dilitha21/Heart-Attack-Risk-Prediction.git
   cd Heart-Attack-Risk-Prediction
   ```

2. **Install Python dependencies:**
   ```sh
   pip install -r requirements.txt
   ```

3. **Train the model:**
   ```sh
   python train_model.py
   ```
   This will generate the model and preprocessing files in the `model/` directory.

4. **Run the Flask API:**
   ```sh
   python api.py
   ```
   The API will be available at `http://localhost:5000`.

5. **Open the frontend:**
   - Open `frontend/index.html` in your browser.
   - Make sure the API is running for predictions to work.

## Field Name Conventions

All form fields, API keys, and model features use the following capitalized names:
- Age
- Gender
- Cholesterol
- Blood Pressure
- Heart Rate
- Smoking
- Obesity
- Diabetes
- Previous Heart Problems
- Medication Use
- Exercise Hours Per Week
- Stress Level

## Example API Request

```
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
```

## Notes
- Ensure all field names are capitalized and match exactly between frontend, backend, and model.
- If you update the dataset or model, retrain using `train_model.py`.
- For any issues, check browser cache and ensure the latest files are loaded.

## License
MIT

