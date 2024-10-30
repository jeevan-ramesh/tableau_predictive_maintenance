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

def predict_failure(type_list, air_temp_list, process_temp_list, rotational_speed_list, torque_list, tool_wear_list):
    """
    Predict machine failure based on input lists.
    
    Args:
        type_list: List of tool types
        air_temp_list: List of air temperatures
        process_temp_list: List of process temperatures
        rotational_speed_list: List of rotational speeds
        torque_list: List of torques
        tool_wear_list: List of tool wear times
        
    Returns:
        Dictionary with predictions and probabilities
    """
    try:
        # Initialize an empty DataFrame for input data
        df = pd.DataFrame({
            'Type': type_list,
            'Air temperature K': air_temp_list,
            'Process temperature K': process_temp_list,
            'Rotational speed rpm': rotational_speed_list,
            'Torque Nm': torque_list,
            'Tool wear min': tool_wear_list
        })
        
        # Validate input data
        validate_input(df)
        
        # Make predictions
        predictions = xgb_model.predict(df)
        probabilities = xgb_model.predict_proba(df)
        
        # Convert numpy arrays to lists
        predictions_list = predictions.tolist()
        probabilities_list = probabilities.tolist()
        predictions_final=[predictions_list,probabilities_list]
        # Prepare response
        response = {
            'predictions': predictions_list,
            'probabilities': probabilities_list,
            'success': True,
            'message': 'Predictions generated successfully'
        }
        
        return predictions_final
        
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
            'Predicts machine failure based on input lists.',
            override=True
        )
        logger.info("Model deployed successfully!")
        
        # Create test data as lists
        test_data = {
            'type_list': [1],
            'air_temp_list': [298.1],
            'process_temp_list': [308.6],
            'rotational_speed_list': [1551],
            'torque_list': [42.8],
            'tool_wear_list': [0]
        }
        
        # Test the deployed endpoint
        test_result = client.query('predict_failure', 
                                    test_data['type_list'], 
                                    test_data['air_temp_list'], 
                                    test_data['process_temp_list'], 
                                    test_data['rotational_speed_list'], 
                                    test_data['torque_list'], 
                                    test_data['tool_wear_list'])
        
        logger.info("Test prediction successful!")
        logger.info(f"Test result: {test_result}")
        
    except Exception as e:
        logger.error(f"Deployment failed: {str(e)}")
        raise

if __name__ == "__main__":
    deploy_model()
