 

# Heart Attack Risk Prediction

This project predicts the risk of a heart attack using machine learning. It includes a trained Random Forest model, data preprocessing tools, and a web-based frontend for user interaction.

## Features
- Predicts heart attack risk based on user input
- Utilizes a Random Forest classifier
- Data preprocessing with scaling and label encoding
- Web frontend for easy access

## Project Structure
- `train_model.py`: Trains the machine learning model and saves preprocessing objects
- `ML_CW.py`: Contains core ML logic and utilities
- `api.py`: Backend API for serving predictions
- `model/`: Stores trained model (`rf_model.pkl`), scaler, and label encoders
- `frontend/`: Web interface (HTML, CSS, JS, assets)
- `heart-attack-risk-prediction-dataset.csv`: Dataset used for training
- `requirements.txt`: Python dependencies

## Usage
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Train the model (if needed):
   ```bash
   python train_model.py
   ```
3. Run the API server:
   ```bash
   python api.py
   ```
4. Open `frontend/index.html` in your browser to use the web app.

## Dataset
The dataset contains medical and demographic features relevant to heart attack risk. See `heart-attack-risk-prediction-dataset.csv` for details.

## License
This project is for educational purposes.

