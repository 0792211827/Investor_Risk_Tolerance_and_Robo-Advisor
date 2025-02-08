import joblib
import pandas as pd
import numpy as np
import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(os.path.join(project_root, "src"))
from feature_preprocessing import plot_log_transformation, standard_scale_features


def load_model(model_path):
    """
    Loads the trained model from the path.
    """
    return joblib.load('/Users/sylviabhoke/Downloads/personal_repos folder/Investor_Risk_Tolerance_and_Robo-Advisor/models/best_model.pkl')

def preprocess_input_data(df, columns_to_log_transform):
    """
    Applies log transformation and standard scaling to input data.
    """
    df = plot_log_transformation(df, columns_to_log_transform)
    df = standard_scale_features(df)
    return df

def make_predictions(input_data, model_path, columns_to_log_transform):
    """
    Takes user input, preprocesses it, and returns predictions.
    """
    model = load_model(model_path)
    
    # Convert input into a DataFrame
    df = pd.DataFrame([input_data])
    df = preprocess_input_data(df, columns_to_log_transform)
    
    predictions = model.predict(df)
    return predictions[0]

if __name__ == "__main__":
    model_path = "best_model.pkl"
    columns_to_log_transform = ['INCOME', 'NETWORTH']
    
    # Example input for testing (Replace with actual user input in Streamlit)
    user_input = {
         'AGE': 30,
         'EDUCATION_LEVEL': 3,
         'MARITAL_STATUS': 1,
         'NO_OF_KIDS': 2,
         'OCCUPATION_CATEGORY': 2,
         'INCOME': 50000,
         'RISK_LEVEL': 3,
         'SPENDING_VS_INCOME': 0.5,
         'SPENDING_LEVEL': 2,
         'NETWORTH': 1000000
    }
    
    prediction = make_predictions(user_input, model_path, columns_to_log_transform)
    print("Predicted Risk Tolerance:", prediction)