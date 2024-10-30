import joblib
import pandas as pd
from tabpy.tabpy_tools.client import Client
import xgboost as xgb
import logging
import numpy as np

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_model(model_path):
    """Load the XGBoost model safely"""
    try:
        return joblib.load(model_path)
    except Exception as e:
        logger.error(f"Failed to load model from {model_path}: {str(e)}")
        raise

def validate_input(data):
    """Validate input data structure and types"""
    required_columns = [
        'Type', 
        'Air temperature K', 
        'Process temperature K', 
        'Rotational speed rpm', 
        'Torque Nm', 
        'Tool wear min'
    ]
    
    # Check if all required columns are present
    for col in required_columns:
        if col not in data.columns:
            raise ValueError(f"Missing required column: {col}")
    
    # Check for null values
    if data.isnull().any().any():
        raise ValueError("Input data contains null values")
    
    # Ensure numeric columns are numeric
    numeric_columns = required_columns[1:]  # All except 'Type'
    for col in numeric_columns:
        if not pd.to_numeric(data[col], errors='coerce').notnull().all():
            raise ValueError(f"Column {col} contains non-numeric values")
    
    return True

def predict_failure(_data):
    """
    Predict machine failure based on input data.
    
    Args:
        _data: Dictionary containing the feature arrays
        
    Returns:
        Dictionary with predictions and probabilities
    """
    try:
        # Convert input data to DataFrame
        if isinstance(_data, dict):
            df = pd.DataFrame(_data)
        else:
            df = pd.DataFrame.from_dict(_data)
        
        # Validate input data
        validate_input(df)
        
        # Make predictions
        predictions = xgb_model.predict(df)
        probabilities = xgb_model.predict_proba(df)
        
        # Convert numpy arrays to lists
        predictions_list = predictions.tolist()
        probabilities_list = probabilities.tolist()
        
        # Prepare response
        response = {
            'predictions': predictions_list,
            'probabilities': probabilities_list,
            'success': True,
            'message': 'Predictions generated successfully'
        }
        
        return response
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return {
            'success': False,
            'error': str(e),
            'message': 'Failed to generate predictions'
        }

def deploy_model():
    """Deploy the model to TabPy with error handling"""
    try:
        # Model path
        model_path = 'D:/network threat detection/model/xgboost_model.pkl'
        
        # Load model
        global xgb_model
        xgb_model = load_model(model_path)
        logger.info("Model loaded successfully")
        
        # Connect to TabPy
        client = Client('http://localhost:9004/')
        logger.info("Connected to TabPy server")
        
        # Deploy the model
        client.deploy(
            'predict_failure',
            predict_failure,
            'Predicts machine failure based on input data.',
            override=True
        )
        logger.info("Model deployed successfully!")
        
        # Create test data as a dictionary
        test_data = {
            'Type': ['L'],
            'Air temperature K': [298.1],
            'Process temperature K': [308.6],
            'Rotational speed rpm': [1551],
            'Torque Nm': [42.8],
            'Tool wear min': [0]
        }
        
        # Test the deployed endpoint
        test_result = client.query('predict_failure', test_data)
        logger.info("Test prediction successful!")
        logger.info(f"Test result: {test_result}")
        
    except Exception as e:
        logger.error(f"Deployment failed: {str(e)}")
        raise

if __name__ == "__main__":
    deploy_model()
    