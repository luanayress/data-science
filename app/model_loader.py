"""
Model Deployment Module
========================
This module handles the saving and loading of trained models for production use.
It includes utilities for model persistence and preprocessing pipeline setup.

Author: Data Science Team
Date: 2026-01-27
"""

import pickle
import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.ensemble import GradientBoostingClassifier
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define paths for model artifacts
# Use current working directory as fallback if __file__ is not reliable
try:
    MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
except:
    MODEL_DIR = os.getcwd()

# Check if we're in the right directory, if not try current working directory
if not os.path.exists(os.path.join(MODEL_DIR, 'models')):
    MODEL_DIR = os.getcwd()
    logger.info(f"Adjusted MODEL_DIR to current working directory: {MODEL_DIR}")

GB_MODEL_PATH = os.path.join(MODEL_DIR, 'models', 'gradient_boosting_model.pkl')
SCALER_STANDARD_PATH = os.path.join(MODEL_DIR, 'models', 'scaler_standard.pkl')
SCALER_MINMAX_PATH = os.path.join(MODEL_DIR, 'models', 'scaler_minmax.pkl')
PREPROCESSING_CONFIG_PATH = os.path.join(MODEL_DIR, 'models', 'preprocessing_config.pkl')

# Create models directory if it doesn't exist
os.makedirs(os.path.dirname(GB_MODEL_PATH), exist_ok=True)

logger.info(f"Model directory: {MODEL_DIR}")
logger.info(f"Model path: {GB_MODEL_PATH}")


class ModelDeployment:
    """
    Handles model deployment including saving, loading, and serving predictions.
    """
    
    def __init__(self):
        """Initialize ModelDeployment instance."""
        self.model = None
        self.scaler_standard = None
        self.scaler_minmax = None
        self.preprocessing_config = None
        self.is_loaded = False
    
    def load_model(self):
        """
        Load the trained Gradient Boosting model from disk.
        
        Returns:
            bool: True if model loaded successfully, False otherwise
        """
        try:
            logger.info(f"Looking for model at: {GB_MODEL_PATH}")
            if os.path.exists(GB_MODEL_PATH):
                self.model = joblib.load(GB_MODEL_PATH)
                logger.info(f"✓ Gradient Boosting model loaded from {GB_MODEL_PATH}")
                return True
            else:
                logger.error(f"✗ Model not found at {GB_MODEL_PATH}")
                logger.error(f"Current working directory: {os.getcwd()}")
                logger.error(f"Files in current directory: {os.listdir(os.getcwd())}")
                if os.path.exists(os.path.join(os.getcwd(), 'models')):
                    logger.error(f"Files in models directory: {os.listdir(os.path.join(os.getcwd(), 'models'))}")
                return False
        except Exception as e:
            logger.error(f"✗ Error loading model: {str(e)}")
            return False
    
    def load_scalers(self):
        """
        Load the preprocessing scalers from disk.
        
        Returns:
            bool: True if scalers loaded successfully, False otherwise
        """
        try:
            if os.path.exists(SCALER_STANDARD_PATH) and os.path.exists(SCALER_MINMAX_PATH):
                self.scaler_standard = joblib.load(SCALER_STANDARD_PATH)
                self.scaler_minmax = joblib.load(SCALER_MINMAX_PATH)
                logger.info("✓ Scalers loaded successfully")
                return True
            else:
                logger.error("✗ Scalers not found")
                return False
        except Exception as e:
            logger.error(f"✗ Error loading scalers: {str(e)}")
            return False
    
    def load_preprocessing_config(self):
        """
        Load preprocessing configuration from disk.
        
        Returns:
            bool: True if config loaded successfully, False otherwise
        """
        try:
            if os.path.exists(PREPROCESSING_CONFIG_PATH):
                self.preprocessing_config = joblib.load(PREPROCESSING_CONFIG_PATH)
                logger.info("✓ Preprocessing configuration loaded")
                return True
            else:
                logger.error("✗ Preprocessing configuration not found")
                return False
        except Exception as e:
            logger.error(f"✗ Error loading preprocessing config: {str(e)}")
            return False
    
    def load_all(self):
        """
        Load all model artifacts (model, scalers, and config).
        
        Returns:
            bool: True if all artifacts loaded successfully, False otherwise
        """
        model_loaded = self.load_model()
        scalers_loaded = self.load_scalers()
        config_loaded = self.load_preprocessing_config()
        
        self.is_loaded = model_loaded and scalers_loaded and config_loaded
        
        if self.is_loaded:
            logger.info("✓ All model artifacts loaded successfully")
        else:
            logger.error("✗ Failed to load some model artifacts")
        
        return self.is_loaded
    
    def save_model(self, model):
        """
        Save the trained Gradient Boosting model to disk.
        
        Args:
            model: The trained model to save
        
        Returns:
            bool: True if model saved successfully, False otherwise
        """
        try:
            joblib.dump(model, GB_MODEL_PATH)
            logger.info(f"✓ Model saved to {GB_MODEL_PATH}")
            return True
        except Exception as e:
            logger.error(f"✗ Error saving model: {str(e)}")
            return False
    
    def save_scalers(self, scaler_standard, scaler_minmax):
        """
        Save the preprocessing scalers to disk.
        
        Args:
            scaler_standard: StandardScaler instance
            scaler_minmax: MinMaxScaler instance
        
        Returns:
            bool: True if scalers saved successfully, False otherwise
        """
        try:
            joblib.dump(scaler_standard, SCALER_STANDARD_PATH)
            joblib.dump(scaler_minmax, SCALER_MINMAX_PATH)
            logger.info("✓ Scalers saved successfully")
            return True
        except Exception as e:
            logger.error(f"✗ Error saving scalers: {str(e)}")
            return False
    
    def save_preprocessing_config(self, config):
        """
        Save preprocessing configuration to disk.
        
        Args:
            config (dict): Preprocessing configuration dictionary
        
        Returns:
            bool: True if config saved successfully, False otherwise
        """
        try:
            joblib.dump(config, PREPROCESSING_CONFIG_PATH)
            logger.info("✓ Preprocessing configuration saved")
            return True
        except Exception as e:
            logger.error(f"✗ Error saving preprocessing config: {str(e)}")
            return False
    
    def predict(self, input_data):
        """
        Make predictions on new data using the loaded model.
        
        Args:
            input_data (pd.DataFrame): Input features for prediction
        
        Returns:
            dict: Prediction results including probability and class
        """
        if not self.is_loaded:
            logger.error("✗ Model artifacts not loaded. Call load_all() first.")
            return None
        
        try:
            # Convert to numpy array to avoid feature name mismatch warnings
            if isinstance(input_data, pd.DataFrame):
                input_array = input_data.values
            else:
                input_array = input_data
            
            # Make predictions
            prediction = self.model.predict(input_array)[0]
            prediction_proba = self.model.predict_proba(input_array)[0]
            
            result = {
                'prediction': int(prediction),
                'prediction_label': 'Churned' if prediction == 1 else 'Retained',
                'probability_retained': float(prediction_proba[0]),
                'probability_churned': float(prediction_proba[1]),
                'confidence': float(max(prediction_proba))
            }
            
            return result
        except Exception as e:
            logger.error(f"✗ Error making prediction: {str(e)}")
            return None
    
    def preprocess_input(self, data_dict):
        """
        Preprocess input data for model prediction.
        
        Args:
            data_dict (dict): Dictionary containing input features
        
        Returns:
            pd.DataFrame: Preprocessed features ready for model
        """
        try:
            # Create DataFrame from input with correct order and all features
            df = pd.DataFrame([{
                'NumOfProducts': data_dict.get('NumOfProducts', 0),
                'Age_Squared_StandardScaled': data_dict.get('Age_Squared_StandardScaled', 0),
                'Age_Tenure_Interaction_MinMaxScaled': data_dict.get('Age_Tenure_Interaction_MinMaxScaled', 0)
            }])
            
            # Apply scaling to entire feature set at once
            if self.scaler_standard is not None:
                try:
                    # The scaler was fit on all features together
                    scaled_array = self.scaler_standard.transform(df)
                    df = pd.DataFrame(scaled_array, columns=['NumOfProducts', 'Age_Squared_StandardScaled', 'Age_Tenure_Interaction_MinMaxScaled'])
                except Exception as scale_error:
                    logger.warning(f"Could not apply standard scaler, using raw values: {str(scale_error)}")
            
            logger.info("✓ Input data preprocessed successfully")
            return df
        except Exception as e:
            logger.error(f"✗ Error preprocessing input: {str(e)}")
            return None


def get_deployment_instance():
    """
    Factory function to get a ModelDeployment instance with loaded artifacts.
    
    Returns:
        ModelDeployment: Instance with all artifacts loaded
    """
    deployment = ModelDeployment()
    deployment.load_all()
    return deployment


if __name__ == "__main__":
    # Test model deployment
    print("\n" + "="*70)
    print("MODEL DEPLOYMENT TEST")
    print("="*70)
    
    # Initialize deployment
    deployment = ModelDeployment()
    
    # Try to load artifacts
    if deployment.load_all():
        print("\n✓ All model artifacts loaded successfully!")
        print(f"  Model type: {type(deployment.model).__name__}")
        print(f"  Scalers available: Standard={deployment.scaler_standard is not None}, "
              f"MinMax={deployment.scaler_minmax is not None}")
    else:
        print("\n✗ Model artifacts not found. Please run feature_eng.ipynb and modeling.ipynb first.")
        print("  Then save the models using the save functions.")
    
    print("\n" + "="*70)
