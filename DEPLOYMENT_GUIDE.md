# Customer Churn Prediction - Deployment Guide
## Model Deployment & Streamlit Dashboard Setup

---

## 📋 Table of Contents
1. [Overview](#overview)
2. [Project Structure](#project-structure)
3. [Prerequisites](#prerequisites)
4. [Step-by-Step Setup](#step-by-step-setup)
5. [Running the Dashboard](#running-the-dashboard)
6. [Model Artifacts](#model-artifacts)
7. [File Descriptions](#file-descriptions)
8. [Troubleshooting](#troubleshooting)
9. [Production Deployment](#production-deployment)

---

## 🎯 Overview

This deployment package includes:
- **model_deployment.py** - Model loading and prediction engine
- **app.py** - Interactive Streamlit dashboard
- **save_models.py** - Script to save trained models
- **requirements.txt** - Python dependencies

The system uses a trained Gradient Boosting model to predict customer churn in real-time.

---

## 📁 Project Structure

```
Data Science/
├── exploratory_analysis.ipynb         # EDA notebook
├── feature_eng.ipynb                  # Feature engineering notebook
├── modeling.ipynb                     # Model training notebook
├── model_deployment.py                # Model deployment module (NEW)
├── app.py                             # Streamlit dashboard (NEW)
├── save_models.py                     # Model persistence script (NEW)
├── requirements.txt                   # Dependencies (NEW)
├── DEPLOYMENT_GUIDE.md                # This file (NEW)
├── models/                            # Model artifacts directory (NEW)
│   ├── gradient_boosting_model.pkl
│   ├── scaler_standard.pkl
│   ├── scaler_minmax.pkl
│   └── preprocessing_config.pkl
├── Customer-Churn-Records.csv         # Raw data
├── Customer_Churn_Final_Features.csv  # Engineered features
├── Feature_Engineering_Report.txt     # Summary report
└── README.md                          # Project documentation
```

---

## 🔧 Prerequisites

### System Requirements
- Python 3.7 or higher (3.13.0 tested)
- pip (Python package manager)
- Windows/Linux/macOS

### Python Packages
All required packages are listed in `requirements.txt`

Key packages:
- **streamlit** - Dashboard framework
- **scikit-learn** - Machine learning models
- **pandas** - Data manipulation
- **numpy** - Numerical computing
- **matplotlib/seaborn** - Visualization
- **joblib** - Model serialization

---

## 📝 Step-by-Step Setup

### STEP 1: Ensure Models Are Trained
Run all notebooks in this order:

```bash
# Terminal/Command Prompt
cd "c:\Users\Luan\Desktop\Data Science"

# Run exploratory analysis
jupyter notebook exploratory_analysis.ipynb

# Run feature engineering
jupyter notebook feature_eng.ipynb

# Run model training
jupyter notebook modeling.ipynb
```

### STEP 2: Save Model Artifacts

After running `modeling.ipynb`, the trained models exist in memory. 
Add this code to a new cell at the end of `modeling.ipynb`:

```python
# ====== Save Models for Deployment ======
import joblib
import os

# Create preprocessing config (keep in memory from modeling.ipynb)
preprocessing_config = {
    'features_used': features_to_use,  # From modeling.ipynb
    'scaler_type': 'StandardScaler + MinMaxScaler',
    'model_type': 'GradientBoostingClassifier',
    'training_samples': len(X_train),
    'test_samples': len(X_test),
    'model_performance': {
        'accuracy': accuracy_score(y_test, y_pred_gb),
        'precision': precision_score(y_test, y_pred_gb),
        'recall': recall_score(y_test, y_pred_gb),
        'f1_score': f1_score(y_test, y_pred_gb),
        'roc_auc': roc_auc_score(y_test, y_pred_proba_gb)
    }
}

# Save the models
MODELS_DIR = os.path.join(os.getcwd(), 'models')
os.makedirs(MODELS_DIR, exist_ok=True)

joblib.dump(gb_model, os.path.join(MODELS_DIR, 'gradient_boosting_model.pkl'))
joblib.dump(scaler_standard, os.path.join(MODELS_DIR, 'scaler_standard.pkl'))
joblib.dump(scaler_minmax, os.path.join(MODELS_DIR, 'scaler_minmax.pkl'))
joblib.dump(preprocessing_config, os.path.join(MODELS_DIR, 'preprocessing_config.pkl'))

print("✓ All model artifacts saved successfully!")
print(f"  Location: {MODELS_DIR}")
```

**Output should show:**
```
✓ All model artifacts saved successfully!
  Location: c:\Users\Luan\Desktop\Data Science\models
```

### STEP 3: Install Dependencies

```bash
# Terminal/Command Prompt
cd "c:\Users\Luan\Desktop\Data Science"

# Install all required packages
pip install -r requirements.txt

# Or manually install key packages
pip install streamlit scikit-learn pandas numpy matplotlib seaborn joblib
```

### STEP 4: Verify Model Artifacts

```bash
# Check that models directory was created
dir models
```

You should see:
```
Volume in drive C is Windows
 Directory of c:\Users\Luan\Desktop\Data Science\models

01/27/2026  10:30 AM    <DIR>          .
01/27/2026  10:30 AM    <DIR>          ..
01/27/2026  10:30 AM         1,234,567 gradient_boosting_model.pkl
01/27/2026  10:30 AM            45,678 scaler_standard.pkl
01/27/2026  10:30 AM            45,678 scaler_minmax.pkl
01/27/2026  10:30 AM             5,432 preprocessing_config.pkl
```

---

## 🚀 Running the Dashboard

### Launch the Streamlit Dashboard

```bash
# Terminal/Command Prompt
cd "c:\Users\Luan\Desktop\Data Science"

# Run the dashboard
streamlit run app.py
```

**Expected output:**
```
  You can now view your Streamlit app in your browser.

  Local URL: http://localhost:8501
  Network URL: http://192.168.x.x:8501
```

### Access the Dashboard

Open your web browser and go to:
```
http://localhost:8501
```

---

## 📊 Dashboard Features

### Main Page (Prediction Tab)
- **Input Form**: Enter customer data in sidebar
  - Number of Products (1-4)
  - Age (18-100 years)
  - Tenure (0-10 months)
- **Prediction Results**:
  - Prediction label (Churned/Retained)
  - Churn probability (%)
  - Retention probability (%)
  - Confidence score
  - Visual probability gauge

### Analytics Tab
- **Model Performance Metrics**:
  - Accuracy, Precision, Recall, F1-Score, ROC-AUC
  - Bar chart visualization
- **Model Comparison**:
  - Comparison of all 4 trained models
  - F1-Score comparison chart

### About Tab
- Model overview and how it works
- Dataset information
- Feature descriptions
- Disclaimer and usage notes

---

## 🔧 Model Artifacts

### What Gets Saved?

1. **gradient_boosting_model.pkl** (~1.2 MB)
   - Trained Gradient Boosting classifier
   - Used for making predictions
   - Loaded into memory when dashboard starts

2. **scaler_standard.pkl** (~45 KB)
   - StandardScaler for feature normalization
   - Transforms features to mean=0, std=1
   - Applied to specific engineered features

3. **scaler_minmax.pkl** (~45 KB)
   - MinMaxScaler for feature normalization
   - Transforms features to range [0,1]
   - Applied to specific engineered features

4. **preprocessing_config.pkl** (~5 KB)
   - Configuration metadata
   - Features list, model type, performance metrics
   - Used for validation and documentation

### Why Save These?

- **Consistency**: Same preprocessing applied during training and prediction
- **Reproducibility**: Models are deterministic (random_state=42)
- **Performance**: Scalers calibrated on training data
- **Production Readiness**: Can deploy without retraining

---

## 📄 File Descriptions

### model_deployment.py
```
Purpose: Handles model loading and prediction logic
Key Classes:
  - ModelDeployment: Main class for model operations
    - load_model(): Load trained model
    - load_scalers(): Load preprocessing scalers
    - load_all(): Load all artifacts
    - predict(): Make predictions on new data
    - preprocess_input(): Process user input

Usage:
  from model_deployment import ModelDeployment
  deployment = ModelDeployment()
  deployment.load_all()
  result = deployment.predict(data)
```

### app.py
```
Purpose: Streamlit dashboard application
Key Sections:
  - Page configuration and styling
  - Input form for customer data
  - Prediction display and visualization
  - Model performance analytics
  - About/Help section

Features:
  - Real-time predictions
  - Interactive widgets
  - Performance visualizations
  - Prediction history tracking
  - Session state management

Run with: streamlit run app.py
```

### save_models.py
```
Purpose: Save trained models from notebook to disk
Key Functions:
  - save_model_artifacts(): Save all model artifacts
  - verify_artifacts(): Check if all files were saved correctly
  - create_sample_input(): Generate test data

Usage:
  from save_models import save_model_artifacts
  save_model_artifacts(gb_model, scaler_standard, scaler_minmax, config)
```

### requirements.txt
```
Purpose: List all Python dependencies with versions
Install with: pip install -r requirements.txt

Key packages:
  - streamlit: Web dashboard
  - scikit-learn: ML models
  - pandas: Data manipulation
  - matplotlib/seaborn: Visualization
  - joblib: Model serialization
```

---

## 🐛 Troubleshooting

### Issue: "Model artifacts not loaded"
**Cause:** Model files not saved yet
**Solution:**
```bash
1. Run modeling.ipynb completely
2. Add model saving code (see Step 2 above)
3. Verify models/ directory exists with .pkl files
4. Run: streamlit run app.py
```

### Issue: "ModuleNotFoundError: No module named 'streamlit'"
**Cause:** Dependencies not installed
**Solution:**
```bash
pip install -r requirements.txt
# or
pip install streamlit scikit-learn pandas numpy matplotlib seaborn
```

### Issue: "Port 8501 already in use"
**Cause:** Dashboard already running or port occupied
**Solution:**
```bash
# Use different port
streamlit run app.py --server.port 8502

# Or kill existing process (Windows)
netstat -ano | findstr :8501
taskkill /PID <PID> /F
```

### Issue: "FileNotFoundError: Customer_Churn_Final_Features.csv"
**Cause:** Working directory incorrect
**Solution:**
```bash
# Ensure you're in the correct directory
cd "c:\Users\Luan\Desktop\Data Science"

# Then run
streamlit run app.py
```

### Issue: Predictions show all zeros or NaN
**Cause:** Preprocessing error or model corruption
**Solution:**
```bash
1. Verify scaler files aren't corrupted
2. Retrain models and resave artifacts
3. Check that feature names match exactly
4. Run verify_artifacts() from save_models.py
```

---

## 🌐 Production Deployment

### Option 1: Local Network Sharing

```bash
# Run dashboard accessible from other computers on network
streamlit run app.py --server.address 0.0.0.0
```

Then access from other computers using:
```
http://<your-computer-ip>:8501
```

### Option 2: Cloud Deployment (Streamlit Cloud)

1. Push code to GitHub
2. Connect to Streamlit Cloud
3. Streamlit automatically deploys

**Steps:**
```bash
# Create GitHub repo with files
git init
git add .
git commit -m "Initial commit"
git push origin main

# Then on Streamlit Cloud dashboard
# Connect repo and deploy
```

### Option 3: Docker Containerization

**Create Dockerfile:**
```dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "app.py"]
```

**Build and run:**
```bash
docker build -t churn-prediction .
docker run -p 8501:8501 churn-prediction
```

### Option 4: API Deployment (FastAPI + Flask)

For API-based predictions without GUI:
```python
# Example with Flask
from flask import Flask, jsonify, request
from model_deployment import ModelDeployment

app = Flask(__name__)
deployment = ModelDeployment()
deployment.load_all()

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    result = deployment.predict(data)
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=False)
```

---

## ✅ Checklist for Deployment

- [ ] All notebooks executed successfully
- [ ] Models trained and saved to models/ directory
- [ ] requirements.txt installed
- [ ] models/ directory contains 4 .pkl files
- [ ] model_deployment.py in project directory
- [ ] app.py in project directory
- [ ] Streamlit runs without errors
- [ ] Dashboard loads in browser
- [ ] Predictions working correctly
- [ ] Model artifacts verified with verify_artifacts()

---

## 📞 Support & Maintenance

### Regular Maintenance
- Monitor model performance metrics
- Retrain models periodically (monthly/quarterly)
- Update features based on business changes
- Track prediction accuracy in production

### Model Retraining
```bash
# When new data is available
1. Add new data to Customer-Churn-Records.csv
2. Rerun feature_eng.ipynb
3. Rerun modeling.ipynb
4. Save new models (see Step 2)
5. Restart dashboard: streamlit run app.py
```

### Performance Monitoring
- Track churn predictions vs actual outcomes
- Monitor model accuracy on new data
- Set up alerts for accuracy drops
- Document changes and improvements

---

## 📚 Additional Resources

- [Streamlit Documentation](https://docs.streamlit.io/)
- [scikit-learn Documentation](https://scikit-learn.org/)
- [Pandas Documentation](https://pandas.pydata.org/)
- [Python Official Documentation](https://www.python.org/doc/)

---

**Last Updated:** January 27, 2026
**Status:** Production Ready
**Version:** 1.0
