import streamlit as st
import pandas as pd
import joblib
import numpy as np
import sys
import os
import matplotlib.pyplot as plt
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import risk_models, expected_returns


project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(project_root)
from src.feature_preprocessing.feature_transformation import plot_log_transformation,standard_scale_features
from src.portfolio_optimization.markowitz import calculate_portfolio_statistics, optimize_portfolio


def load_model(model_path):
    """
    Loads the trained model from the specified path.
    """
    return joblib.load(model_path)

def preprocess_input_data(df, columns_to_log_transform):
    df = plot_log_transformation(df, columns_to_log_transform)
    df = standard_scale_features(df)
    return df

# Load stock data
stock_data = pd.read_csv(os.path.join(project_root, "data", "raw", "stock_data.csv"), index_col='Date')

st.title("Investor Risk Tolerance & Robo-Advisor")

st.markdown("This app is a simple robo-advisor that uses a machine learning model to predict the risk tolerance of an investor based on their responses to a questionnaire. The app then uses the predicted risk tolerance to optimize a portfolio allocation using the Markowitz Portfolio Optimization model.")

#sidebar
# Header for the sidebar
st.sidebar.header("Investor Risk Tolerance Questionnaire")

def user_input_features():
    """
    Collects user input for the questionnaire.
    """
    
    age = st.sidebar.slider("Age", min_value=18, max_value=100, value=30)
    education_level = st.sidebar.slider("Education Level", ["High School", "Bachelor's", "Master's", "PhD"])
    marital_status = st.sidebar.slider("Marital Status", ["Single", "Married", "Divorced", "Widowed"])
    no_of_kids = st.sidebar.number_input("Number of Kids", min_value=0, max_value=10, value=2)
    occupation_category = st.sidebar.slider("Occupation Category", ["Student", "Employed", "Self-employed", "Retired"])
    income = st.sidebar.number_input("Annual Income", min_value=0, value=50000)
    net_worth = st.sidebar.number_input("Net Worth", min_value=0, value=1000000)
    spending_vs_income = st.sidebar.slider("Spending vs Income Ratio", 0.0, 1.0, 0.5)
    spending_level = st.sidebar.selectbox("Spending Level", ["Low", "Moderate", "High"])
    risk_level = st.sidebar.slider("Risk Level (1-5)", 1, 5, 3)
    return age, income, net_worth, no_of_kids, spending_vs_income, risk_level, education_level, marital_status, occupation_category, spending_level
