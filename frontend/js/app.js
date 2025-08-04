// API Configuration
const API_BASE_URL = 'http://localhost:5000';

// DOM Elements
let form;
let predictBtn;
let spinner;
let resultsContainer;
let errorContainer;

// Initialize the app when DOM is loaded
document.addEventListener('DOMContentLoaded', function() {
    initializeApp();
});

function initializeApp() {
    // Get DOM elements
    form = document.getElementById('prediction-form');
    predictBtn = document.getElementById('predict-btn');
    spinner = document.getElementById('spinner');
    resultsContainer = document.getElementById('results-container');
    errorContainer = document.getElementById('error-container');
    
    // Attach event listeners
    form.addEventListener('submit', handleFormSubmit);
    form.addEventListener('reset', handleFormReset);
    
    // Initialize stress level slider display
    initializeStressSlider();
    
    console.log('Heart Attack Risk Prediction App initialized');
}

function initializeStressSlider() {
    const stressSlider = document.getElementById('stress');
    const stressValue = document.getElementById('stress-value');
    
    if (stressSlider && stressValue) {
        stressSlider.addEventListener('input', function() {
            stressValue.textContent = this.value;
        });
    }
}

function handleFormSubmit(event) {
    event.preventDefault();
    
    // Hide previous results and errors
    hideResults();
    hideError();
    
    // Collect form data
    const formData = collectFormData();
    
    if (!formData) {
        showError('Please fill in all required fields correctly.');
        return;
    }
    
    // Show loading state
    setLoadingState(true);
    
    // Make prediction request
    makePrediction(formData);
}

function handleFormReset() {
    hideResults();
    hideError();
    
    // Reset stress slider display
    const stressValue = document.getElementById('stress-value');
    if (stressValue) {
        stressValue.textContent = '5';
    }
}

function collectFormData() {
    try {
        // Get all form elements using correct IDs/names as in HTML
        const age = parseInt(document.getElementById('Age').value);
        const gender = document.getElementById('Gender').value;
        const cholesterol = parseInt(document.getElementById('Cholesterol').value);
        const bloodPressure = document.getElementById('Blood Pressure').value;
        const heartRate = parseInt(document.getElementById('Heart Rate').value);
        const smoking = document.getElementById('Smoking').value;
        const exercise = parseFloat(document.getElementById('Exercise Hours Per Week').value);
        const stress = parseInt(document.getElementById('Stress Level').value);
        const obesity = document.getElementById('Obesity').value;
        const diabetes = document.getElementById('Diabetes').value;
        const heartProblems = document.getElementById('Previous Heart Problems').value;
        const medication = document.getElementById('Medication Use').value;

        // Validate required fields
        if (!age || !gender || !cholesterol || !bloodPressure || !heartRate || 
            !smoking || exercise === '' || !stress || !obesity || !diabetes || 
            !heartProblems || !medication) {
            return null;
        }

        // Validate blood pressure format
        if (!bloodPressure.match(/^\d{2,3}\/\d{2,3}$/)) {
            showError('Please enter blood pressure in format: 120/80');
            return null;
        }

        // Construct data object matching API expectations
        const data = {
            'Age': age,
            'Gender': gender,
            'Cholesterol': cholesterol,
            'Blood Pressure': bloodPressure,
            'Heart Rate': heartRate,
            'Smoking': smoking,
            'Obesity': obesity,
            'Diabetes': diabetes,
            'Previous Heart Problems': heartProblems,
            'Medication Use': medication,
            'Exercise Hours Per Week': exercise,
            'Stress Level': stress
        };

        console.log('Collected form data:', data);
        return data;

    } catch (error) {
        console.error('Error collecting form data:', error);
        return null;
    }
}

async function makePrediction(data) {
    try {
        console.log('Making prediction request to:', `${API_BASE_URL}/predict`);
        
        const response = await fetch(`${API_BASE_URL}/predict`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(data)
        });
        
        const result = await response.json();
        
        if (!response.ok) {
            throw new Error(result.error || `HTTP error! status: ${response.status}`);
        }
        
        console.log('Prediction result:', result);
        displayResults(result);
        
    } catch (error) {
        console.error('Prediction error:', error);
        
        if (error.name === 'TypeError' && error.message.includes('fetch')) {
            showError('Unable to connect to the prediction server. Please make sure the API is running on http://localhost:5000');
        } else {
            showError(`Prediction failed: ${error.message}`);
        }
    } finally {
        setLoadingState(false);
    }
}

function displayResults(result) {
    // Get result elements
    const riskLevel = document.getElementById('risk-level');
    const riskPercentage = document.getElementById('risk-percentage');
    const riskDetails = document.getElementById('risk-details');
    
    if (!riskLevel || !riskPercentage || !riskDetails) {
        console.error('Result display elements not found');
        return;
    }
    
    // Display risk level
    const risk = result.risk;
    const riskLabel = risk === 1 ? 'HIGH RISK' : 'LOW RISK';
    const riskClass = risk === 1 ? 'high-risk' : 'low-risk';
    
    riskLevel.textContent = riskLabel;
    riskLevel.className = `risk-level ${riskClass}`;
    
    // Display confidence percentage
    const confidence = Math.round((result.confidence || 0) * 100);
    riskPercentage.textContent = `Confidence: ${confidence}%`;
    
    // Display detailed probabilities if available
    if (result.probabilities) {
        const lowRiskProb = Math.round(result.probabilities.low_risk_probability * 100);
        const highRiskProb = Math.round(result.probabilities.high_risk_probability * 100);
        
        riskDetails.innerHTML = `
            <h4>Detailed Probabilities:</h4>
            <div style="margin: 10px 0;">
                <div style="display: flex; justify-content: space-between; margin: 5px 0;">
                    <span>Low Risk:</span>
                    <strong style="color: #28a745;">${lowRiskProb}%</strong>
                </div>
                <div style="display: flex; justify-content: space-between; margin: 5px 0;">
                    <span>High Risk:</span>
                    <strong style="color: #dc3545;">${highRiskProb}%</strong>
                </div>
            </div>
            ${result.timestamp ? `<small style="color: #6c757d;">Prediction made: ${new Date(result.timestamp).toLocaleString()}</small>` : ''}
        `;
    } else {
        riskDetails.innerHTML = `
            <p>Risk Level: <strong>${riskLabel}</strong></p>
            ${result.timestamp ? `<small style="color: #6c757d;">Prediction made: ${new Date(result.timestamp).toLocaleString()}</small>` : ''}
        `;
    }
    
    // Show results container
    resultsContainer.classList.remove('hidden');
    
    // Scroll to results
    resultsContainer.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

function setLoadingState(loading) {
    if (loading) {
        predictBtn.disabled = true;
        predictBtn.classList.add('loading');
        predictBtn.querySelector('.btn-text').textContent = 'Predicting...';
    } else {
        predictBtn.disabled = false;
        predictBtn.classList.remove('loading');
        predictBtn.querySelector('.btn-text').textContent = 'Predict Risk';
    }
}

function showError(message) {
    const errorMessage = document.getElementById('error-message');
    if (errorMessage) {
        errorMessage.textContent = message;
        errorContainer.classList.remove('hidden');
        
        // Scroll to error
        errorContainer.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }
    
    console.error('Error:', message);
}

function hideError() {
    if (errorContainer) {
        errorContainer.classList.add('hidden');
    }
}

function hideResults() {
    if (resultsContainer) {
        resultsContainer.classList.add('hidden');
    }
}

// Global function for error close button
window.hideError = hideError;

// Test API connection on load
document.addEventListener('DOMContentLoaded', function() {
    setTimeout(testAPIConnection, 1000);
});

async function testAPIConnection() {
    try {
        const response = await fetch(`${API_BASE_URL}/`);
        if (response.ok) {
            console.log('✅ API connection successful');
        } else {
            console.warn('⚠️ API responded but may have issues');
        }
    } catch (error) {
        console.warn('⚠️ API connection failed:', error.message);
        console.log('Make sure to run: python api.py');
    }
}