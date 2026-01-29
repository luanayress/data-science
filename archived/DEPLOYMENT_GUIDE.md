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

This professional ML project structure includes:
- **app/** - Streamlit dashboard and API serving layer
- **src/** - Core ML code (data, features, models, pipelines, utilities)
- **notebooks/** - Exploratory analysis and prototyping
- **models/** - Versioned model artifacts
- **tests/** - Automated unit tests
- **configs/** - Configuration management
- **.github/workflows/** - CI/CD pipelines
- **Docker support** - Container deployment

The system uses a trained Gradient Boosting model to predict customer churn in real-time.

---

## 📁 Project Structure

```
Data Science/
│
├── app/                         # Serving / API / Dashboard
│   ├── app.py                   # Streamlit dashboard
│   ├── model_loader.py          # Model loading utilities
│   ├── schema.py                # Data validation schemas
│   └── __init__.py
│
├── src/                         # ML Core Code
│   ├── data/
│   │   ├── load_data.py         # Data loading utilities
│   │   ├── validation.py        # Data validation logic
│   │   └── split.py             # Train/test splitting
│   │
│   ├── features/
│   │   ├── build_features.py    # Feature engineering
│   │   └── transformers.py      # Custom transformers
│   │
│   ├── models/
│   │   ├── train.py             # Model training logic
│   │   ├── evaluate.py          # Model evaluation
│   │   ├── predict.py           # Prediction functions
│   │   └── registry.py          # Model registry/tracking
│   │
│   ├── pipelines/
│   │   ├── training_pipeline.py # Training workflow
│   │   └── inference_pipeline.py# Inference workflow
│   │
│   └── utils/
│       ├── config.py            # Configuration management
│       ├── logger.py            # Logging utilities
│       └── paths.py             # Path utilities
│
├── notebooks/                   # Exploration Only
│   ├── 01_eda.ipynb             # Exploratory data analysis
│   ├── 02_feature_engineering.ipynb # Feature engineering
│   └── 03_modeling.ipynb        # Model development
│
├── models/                      # Model Artifacts (Versioned)
│   └── v1/
│       ├── model.pkl            # Trained model
│       ├── scaler.pkl           # Feature scaler
│       └── metadata.json        # Model metadata
│
├── data/                        # Data Directory
│   ├── raw/                     # Raw data
│   ├── processed/               # Processed features
│   └── external/                # External data sources
│
├── tests/                       # Automated Tests
│   ├── test_features.py
│   ├── test_model.py
│   └── test_api.py
│
├── configs/                     # Configuration Files
│   ├── training.yaml            # Training config
│   ├── inference.yaml           # Inference config
│   └── features.yaml            # Feature config
│
├── .github/workflows/           # CI/CD Pipelines
│   ├── ci.yml                   # Continuous integration
│   └── deploy.yml               # Deployment pipeline
│
├── Dockerfile                   # Docker containerization
├── requirements.txt             # Python dependencies
├── pyproject.toml              # Project metadata
├── Makefile                     # Build automation
├── README.md                    # Project documentation
└── DEPLOYMENT_GUIDE.md          # This file
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
jupyter notebook notebooks/01_eda.ipynb

# Run feature engineering
jupyter notebook notebooks/02_feature_engineering.ipynb

# Run model training
jupyter notebook notebooks/03_modeling.ipynb
```

### STEP 2: Save Model Artifacts

After running `notebooks/03_modeling.ipynb`, the trained models exist in memory. 
Add this code to a new cell at the end of `notebooks/03_modeling.ipynb`:

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

# Save the models to versioned directory
MODELS_DIR = os.path.join(os.getcwd(), 'models', 'v1')
os.makedirs(MODELS_DIR, exist_ok=True)

joblib.dump(gb_model, os.path.join(MODELS_DIR, 'model.pkl'))
joblib.dump(scaler_standard, os.path.join(MODELS_DIR, 'scaler.pkl'))
joblib.dump(preprocessing_config, os.path.join(MODELS_DIR, 'metadata.json'))

print("✓ All model artifacts saved successfully!")
print(f"  Location: {MODELS_DIR}")
```

**Output should show:**
```
✓ All model artifacts saved successfully!
  Location: c:\Users\Luan\Desktop\Data Science\models\v1
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
# Check that models/v1 directory was created
dir models\v1
```

You should see:
```
Volume in drive C is Windows
 Directory of c:\Users\Luan\Desktop\Data Science\models\v1

01/27/2026  10:30 AM    <DIR>          .
01/27/2026  10:30 AM    <DIR>          ..
01/27/2026  10:30 AM         1,234,567 model.pkl
01/27/2026  10:30 AM            45,678 scaler.pkl
01/27/2026  10:30 AM             2,345 metadata.json
```

---

## 🚀 Running the Dashboard

### Launch the Streamlit Dashboard

```bash
# Terminal/Command Prompt
cd "c:\Users\Luan\Desktop\Data Science"

# Run the dashboard
streamlit run app/app.py
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

1. **model.pkl** (~1.2 MB)
   - Trained Gradient Boosting classifier
   - Used for making predictions
   - Loaded into memory when dashboard starts
   - Location: `models/v1/model.pkl`

2. **scaler.pkl** (~45 KB)
   - StandardScaler for feature normalization
   - Transforms features to mean=0, std=1
   - Applied to specific engineered features
   - Location: `models/v1/scaler.pkl`

3. **metadata.json** (~5 KB)
   - Configuration metadata
   - Features list, model type, performance metrics
   - Used for validation and documentation
   - Location: `models/v1/metadata.json`

### Why Use Versioned Models?

- **Version Control**: Track model changes over time (v1, v2, etc.)
- **Rollback Capability**: Revert to previous models if needed
- **A/B Testing**: Compare different model versions in production
- **Reproducibility**: Keep historical model artifacts
- **Consistency**: Same preprocessing applied during training and prediction
- **Production Readiness**: Can deploy without retraining

---

## 📄 File Descriptions

### app/app.py
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

Run with: streamlit run app/app.py
```

### app/model_loader.py
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
  from app.model_loader import ModelDeployment
  deployment = ModelDeployment()
  deployment.load_all()
  result = deployment.predict(data)
```

### app/schema.py
```
Purpose: Data validation and schema definitions
Key Classes:
  - CustomerData: Pydantic model for input validation
  - PredictionResponse: Response schema

Features:
  - Type validation
  - Data constraints
  - Serialization/deserialization
```

### src/data/load_data.py
```
Purpose: Data loading utilities
Key Functions:
  - load_raw_data(): Load raw CSV files
  - load_processed_data(): Load engineered features

Usage:
  from src.data.load_data import load_raw_data
  df = load_raw_data('data/raw/Customer-Churn-Records.csv')
```

### src/features/build_features.py
```
Purpose: Feature engineering pipeline
Key Functions:
  - build_features(): Apply all transformations
  - create_new_features(): Generate derived features

Usage:
  from src.features.build_features import build_features
  X_transformed = build_features(X_raw)
```

### src/models/train.py
```
Purpose: Model training logic
Key Functions:
  - train_model(): Train classifier
  - cross_validate(): Evaluate with CV

Usage:
  from src.models.train import train_model
  model = train_model(X_train, y_train)
```

### src/models/evaluate.py
```
Purpose: Model evaluation metrics
Key Functions:
  - evaluate_model(): Calculate performance metrics
  - plot_confusion_matrix(): Visualization

Usage:
  from src.models.evaluate import evaluate_model
  metrics = evaluate_model(model, X_test, y_test)
```

### src/pipelines/training_pipeline.py
```
Purpose: Complete training workflow
Key Functions:
  - run_training_pipeline(): End-to-end training
  - save_pipeline_artifacts(): Persist models

Features:
  - Data loading and preprocessing
  - Feature engineering
  - Model training
  - Evaluation and reporting
```

### src/pipelines/inference_pipeline.py
```
Purpose: Complete inference workflow
Key Functions:
  - run_inference_pipeline(): End-to-end prediction
  - load_pipeline_artifacts(): Load saved models

Features:
  - Model loading
  - Input preprocessing
  - Batch prediction
  - Result formatting
```

### src/utils/config.py
```
Purpose: Configuration management
Key Functions:
  - load_config(): Load YAML configs
  - get_config_value(): Access config settings

Usage:
  from src.utils.config import load_config
  config = load_config('configs/training.yaml')
```

### src/utils/logger.py
```
Purpose: Logging utilities
Key Functions:
  - get_logger(): Get configured logger instance
  - setup_logging(): Initialize logging

Usage:
  from src.utils.logger import get_logger
  logger = get_logger(__name__)
  logger.info('Training started')
```

### notebooks/01_eda.ipynb
```
Purpose: Exploratory Data Analysis
Contents:
  - Data loading and inspection
  - Statistical summaries
  - Visualizations
  - Correlation analysis
  - Missing value analysis
```

### notebooks/02_feature_engineering.ipynb
```
Purpose: Feature engineering exploration
Contents:
  - Feature creation and transformation
  - Scaling and normalization
  - Feature selection
  - Engineering decisions documentation
```

### notebooks/03_modeling.ipynb
```
Purpose: Model development and training
Contents:
  - Multiple model implementations
  - Hyperparameter tuning
  - Model comparison
  - Performance metrics
  - Model selection and final training
```

### configs/training.yaml
```
Purpose: Training configuration
Contains:
  - Model hyperparameters
  - Data paths
  - Train/test split ratio
  - Feature engineering options
  - Validation settings
```

### configs/features.yaml
```
Purpose: Feature engineering configuration
Contains:
  - Feature list
  - Scaling parameters
  - Feature interaction settings
  - Transformation rules
```

### Dockerfile
```
Purpose: Container image specification
Contains:
  - Base Python image
  - Dependency installation
  - Code copying
  - Streamlit port exposure
  - Startup command

Usage:
  docker build -t churn-prediction .
  docker run -p 8501:8501 churn-prediction
```

### requirements.txt
```
Purpose: Python package dependencies
Contains:
  - streamlit: Web dashboard
  - scikit-learn: ML models
  - pandas: Data manipulation
  - numpy: Numerical computing
  - pydantic: Data validation
  - pyyaml: Configuration parsing
  - pytest: Testing framework

Install with: pip install -r requirements.txt
```

### pyproject.toml
```
Purpose: Project metadata and configuration
Contains:
  - Package information
  - Dependencies
  - Build system configuration
  - Tool configurations (pytest, etc.)
```

### Makefile
```
Purpose: Build and development automation
Common targets:
  - make install: Install dependencies
  - make train: Run training pipeline
  - make test: Run test suite
  - make dashboard: Start Streamlit dashboard
  - make docker-build: Build Docker image
  - make docker-run: Run Docker container
```

---

## 🐛 Troubleshooting

### Issue: "Model artifacts not loaded"
**Cause:** Model files not saved yet
**Solution:**
```bash
1. Run notebooks/03_modeling.ipynb completely
2. Add model saving code (see Step 2 above)
3. Verify models/v1 directory exists with .pkl and .json files
4. Run: streamlit run app/app.py
```

### Issue: "ModuleNotFoundError: No module named 'streamlit'"
**Cause:** Dependencies not installed
**Solution:**
```bash
pip install -r requirements.txt
# or
pip install streamlit scikit-learn pandas numpy pydantic pyyaml
```

### Issue: "Port 8501 already in use"
**Cause:** Dashboard already running or port occupied
**Solution:**
```bash
# Use different port
streamlit run app/app.py --server.port 8502

# Or kill existing process (Windows)
netstat -ano | findstr :8501
taskkill /PID <PID> /F
```

### Issue: "FileNotFoundError: data not found"
**Cause:** Working directory incorrect or data paths not in `data/raw/`
**Solution:**
```bash
# Ensure you're in the correct directory
cd "c:\Users\Luan\Desktop\Data Science"

# Verify data files are in correct location
dir data\raw

# Then run
streamlit run app/app.py
```

### Issue: "ModuleNotFoundError: No module named 'src'"
**Cause:** Python path not configured or running from wrong directory
**Solution:**
```bash
# Ensure you're in the project root
cd "c:\Users\Luan\Desktop\Data Science"

# Set PYTHONPATH
set PYTHONPATH=%cd%

# Then run
streamlit run app/app.py
```

### Issue: Predictions show all zeros or NaN
**Cause:** Preprocessing error or model corruption
**Solution:**
```bash
1. Verify scaler files aren't corrupted
2. Check that feature names match exactly
3. Retrain models and resave artifacts (Step 2)
4. Verify metadata.json contains correct config
```

---

## 🌐 Production Deployment

### Option 1: Local Network Sharing

```bash
# Run dashboard accessible from other computers on network
streamlit run app/app.py --server.address 0.0.0.0
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

The Dockerfile is already provided in the project root:

**Build and run:**
```bash
docker build -t churn-prediction .
docker run -p 8501:8501 churn-prediction
```

### Option 4: API Deployment (FastAPI)

For API-based predictions without GUI:
```python
# Example with FastAPI (src/api/main.py)
from fastapi import FastAPI
from pydantic import BaseModel
from app.model_loader import ModelDeployment

app = FastAPI(title="Customer Churn Prediction API")
deployment = ModelDeployment()
deployment.load_all()

class CustomerData(BaseModel):
    num_products: int
    age: int
    tenure: int

@app.post("/predict")
async def predict(customer: CustomerData):
    data = customer.dict()
    result = deployment.predict(data)
    return result

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

**Run API:**
```bash
# Install fastapi and uvicorn
pip install fastapi uvicorn

# Run the API server
python -m uvicorn src.api.main:app --reload
```

---

## ✅ Checklist for Deployment

- [ ] All notebooks executed successfully
- [ ] Models trained and saved to models/v1/ directory
- [ ] requirements.txt installed
- [ ] models/v1/ directory contains model.pkl, scaler.pkl, metadata.json
- [ ] app/ directory contains app.py, model_loader.py, schema.py
- [ ] src/ directory contains data/, features/, models/, pipelines/, utils/
- [ ] Streamlit runs without errors
- [ ] Dashboard loads in browser
- [ ] Predictions working correctly
- [ ] All imports from src/ work correctly
- [ ] Tests pass (if applicable)

---

## 📞 Support & Maintenance

### Regular Maintenance
- Monitor model performance metrics
- Retrain models periodically (monthly/quarterly)
- Update features based on business changes
- Track prediction accuracy in production
- Review logs and error messages

### Model Retraining
```bash
# When new data is available
1. Add new raw data to data/raw/
2. Rerun notebooks/02_feature_engineering.ipynb
3. Rerun notebooks/03_modeling.ipynb
4. Save new models to models/v2/ (see Step 2)
5. Update config to use new model version
6. Restart dashboard: streamlit run app/app.py
```

### Using Makefile for Common Tasks
```bash
# Install dependencies
make install

# Run training pipeline
make train

# Run tests
make test

# Start dashboard
make dashboard

# Build Docker image
make docker-build

# Run Docker container
make docker-run
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
