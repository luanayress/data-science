"""DEPRECATED: legacy v1 artifact writer; use scripts/train_churn.py.

Model Persistence Script
=========================
This script saves the trained models, scalers, and preprocessing configuration
from the modeling.ipynb notebook to disk for deployment.

Instructions:
1. Run modeling.ipynb to train all models
2. Copy this script to the same directory as modeling.ipynb
3. Run this script in a Jupyter cell or Python terminal
4. This will save all artifacts needed for the Streamlit dashboard

Author: Data Science Team
Date: 2026-01-27
"""

import os
import sys
import joblib
import pandas as pd
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define paths
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(CURRENT_DIR, 'models')

# Create models directory if it doesn't exist
os.makedirs(MODELS_DIR, exist_ok=True)

# Define artifact paths
GB_MODEL_PATH = os.path.join(MODELS_DIR, 'gradient_boosting_model.pkl')
SCALER_STANDARD_PATH = os.path.join(MODELS_DIR, 'scaler_standard.pkl')
SCALER_MINMAX_PATH = os.path.join(MODELS_DIR, 'scaler_minmax.pkl')
PREPROCESSING_CONFIG_PATH = os.path.join(MODELS_DIR, 'preprocessing_config.pkl')


def save_model_artifacts(gb_model, scaler_standard, scaler_minmax, preprocessing_config):
    """
    Save all trained model artifacts to disk.
    
    Args:
        gb_model: Trained Gradient Boosting model
        scaler_standard: StandardScaler instance
        scaler_minmax: MinMaxScaler instance
        preprocessing_config (dict): Preprocessing configuration
    
    Returns:
        bool: True if all artifacts saved successfully, False otherwise
    """
    
    print("\n" + "="*70)
    print("SAVING MODEL ARTIFACTS")
    print("="*70)
    
    try:
        # Save Gradient Boosting model
        joblib.dump(gb_model, GB_MODEL_PATH)
        logger.info(f"✓ Gradient Boosting model saved to {GB_MODEL_PATH}")
        print(f"  → Saved: {GB_MODEL_PATH}")
        
        # Save StandardScaler
        joblib.dump(scaler_standard, SCALER_STANDARD_PATH)
        logger.info(f"✓ StandardScaler saved to {SCALER_STANDARD_PATH}")
        print(f"  → Saved: {SCALER_STANDARD_PATH}")
        
        # Save MinMaxScaler
        joblib.dump(scaler_minmax, SCALER_MINMAX_PATH)
        logger.info(f"✓ MinMaxScaler saved to {SCALER_MINMAX_PATH}")
        print(f"  → Saved: {SCALER_MINMAX_PATH}")
        
        # Save preprocessing configuration
        joblib.dump(preprocessing_config, PREPROCESSING_CONFIG_PATH)
        logger.info(f"✓ Preprocessing config saved to {PREPROCESSING_CONFIG_PATH}")
        print(f"  → Saved: {PREPROCESSING_CONFIG_PATH}")
        
        print("\n✓ All model artifacts saved successfully!")
        print(f"  Location: {MODELS_DIR}")
        
        return True
    
    except Exception as e:
        logger.error(f"✗ Error saving model artifacts: {str(e)}")
        print(f"\n✗ Error: {str(e)}")
        return False


def verify_artifacts():
    """
    Verify that all required artifacts exist.
    
    Returns:
        bool: True if all artifacts exist, False otherwise
    """
    print("\n" + "="*70)
    print("VERIFYING MODEL ARTIFACTS")
    print("="*70)
    
    artifacts = {
        'Gradient Boosting Model': GB_MODEL_PATH,
        'StandardScaler': SCALER_STANDARD_PATH,
        'MinMaxScaler': SCALER_MINMAX_PATH,
        'Preprocessing Config': PREPROCESSING_CONFIG_PATH
    }
    
    all_exist = True
    for name, path in artifacts.items():
        if os.path.exists(path):
            size = os.path.getsize(path)
            logger.info(f"✓ {name}: Found ({size:,} bytes)")
            print(f"  ✓ {name}: {path}")
        else:
            logger.error(f"✗ {name}: Not found")
            print(f"  ✗ {name}: {path} (NOT FOUND)")
            all_exist = False
    
    print("\n" + "="*70)
    if all_exist:
        print("✓ All artifacts verified successfully!")
        print("  Dashboard is ready to run: streamlit run app/app.py")
    else:
        print("✗ Some artifacts are missing.")
        print("  Please run save_model_artifacts() with trained models.")
    print("="*70)
    
    return all_exist


def create_sample_input():
    """
    Create a sample input for testing the model.
    
    Returns:
        dict: Sample customer data
    """
    sample_data = {
        'NumOfProducts': 2,
        'Age_Squared_StandardScaled': (35 ** 2),  # Age 35
        'Age_Tenure_Interaction_MinMaxScaled': (35 * 5)  # Tenure 5 months
    }
    return sample_data


# Usage instructions
USAGE_INSTRUCTIONS = """
================================================================================
HOW TO USE THIS SCRIPT
================================================================================

STEP 1: Save Models from Jupyter Notebook
------------------------------------------
Add this code to the last cell of modeling.ipynb:

```python
import joblib
import os

# Get the models and scalers from the trained models in the notebook
# (These are already in memory after running modeling.ipynb)

# Example preprocessing config
preprocessing_config = {
    'features_used': features_to_use,
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

# Save artifacts
exec(open('save_models.py').read())
save_model_artifacts(gb_model, scaler_standard, scaler_minmax, preprocessing_config)
```

STEP 2: Run Streamlit Dashboard
--------------------------------
In terminal/command prompt:

cd "c:\\Users\\Luan\\Desktop\\Data Science"
streamlit run app/app.py

STEP 3: Access the Dashboard
------------------------------
Open your browser to: http://localhost:8501

================================================================================
DETAILED INSTRUCTIONS FOR SAVING MODELS
================================================================================

1. In modeling.ipynb, after training all models, add a cell with:

   # Define preprocessing config
   preprocessing_config = {
       'features_used': features_to_use,
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

2. Then run this Python script with the models loaded:

   from save_models import save_model_artifacts
   save_model_artifacts(gb_model, scaler_standard, scaler_minmax, preprocessing_config)

3. The script will create a 'models' directory and save:
   - gradient_boosting_model.pkl (trained model)
   - scaler_standard.pkl (StandardScaler)
   - scaler_minmax.pkl (MinMaxScaler)
   - preprocessing_config.pkl (configuration)

4. Run the dashboard with:
   streamlit run app/app.py

================================================================================
TROUBLESHOOTING
================================================================================

Q: "Model artifacts not loaded" error
A: Make sure you've saved the models first using the instructions above.
   Check that the 'models' directory exists with all .pkl files.

Q: "Module not found" error
A: Install required packages:
   pip install streamlit scikit-learn pandas numpy matplotlib seaborn

Q: Dashboard runs but predictions fail
A: Verify model artifacts exist in the 'models' directory.
   Run verify_artifacts() to check all files.

================================================================================
"""

if __name__ == "__main__":
    print(USAGE_INSTRUCTIONS)
    print("\n" + "="*70)
    print("SCRIPT READY")
    print("="*70)
    print("\nTo save models, call:")
    print("  save_model_artifacts(gb_model, scaler_standard, scaler_minmax, preprocessing_config)")
    print("\nTo verify saved artifacts, call:")
    print("  verify_artifacts()")
    print("\n" + "="*70)
