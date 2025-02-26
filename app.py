import streamlit as st
import pandas as pd
import joblib
import numpy as np
import sys
import os
from sklearn.preprocessing import LabelEncoder
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import risk_models, expected_returns


project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(project_root)
from src.modeling.predict import make_predictions
from src.feature_preprocessing.feature_transformation import log_transformation,  encode_categorical_features
from src.portfolio_optimization.markowitz import calculate_portfolio_statistics, optimize_portfolio


def load_model(model_path):
    """
    Loads the trained model from the specified path.
    """
    return joblib.load(model_path)



# Load stock data
stock_data = pd.read_csv('/Users/sylviabhoke/Downloads/personal_repos folder/Investor_Risk_Tolerance_and_Robo-Advisor/data/raw/stock_data.csv', index_col='Date')

st.title("Investor Risk Tolerance & Robo-Advisor")

st.markdown("This app is a simple robo-advisor that uses a machine learning model to predict the risk tolerance of an investor based on their responses to a questionnaire. The app then uses the predicted risk tolerance to optimize a portfolio allocation using the Markowitz Portfolio Optimization model.* Step 1: Answer the questionnaire on the sidebar to predict your risk tolerance. * Step 2: Select your preferred portfolio stock/assets for optimization.")

#sidebar
# Header for the sidebar
st.sidebar.header("Investor Risk Tolerance Questionnaire")

def user_input_features():
    """
    Collects user input for the questionnaire.
    """
    
    age = st.sidebar.number_input("Age", min_value=18, max_value=100,value=25)
    education_level = st.sidebar.selectbox("Education Level", ['junior_school', 'high_school', 'Bachelors/degree', 'Masters/phd'])
    marital_status = st.sidebar.selectbox("Marital Status", ['single/divorced', 'married'])
    no_of_kids = st.sidebar.number_input("Number of Kids", min_value=0, max_value=7, value=2)
    occupation_category = st.sidebar.selectbox("Occupation Category", ['Unemployed', 'Junior_level', 'mid_level', 'Senior_level',])
    income = st.sidebar.number_input("Annual Income", min_value=0, value=50000)
    net_worth = st.sidebar.number_input("Net Worth", min_value=0, value=1000000)
    spending_vs_income = st.sidebar.selectbox("Spending vs Income Ratio", ['spends_more', 'spends_averagely', 'spends_less' ])
    spending_level = st.sidebar.selectbox("Spending Level (1-5, 5 being high)", [1, 2, 3, 4, 5])
    risk_level = st.sidebar.slider("willingness to a Risk  (1-4)", 1, 4, 3)
    


 # **Store input in a dictionary**
    user_data = {
    "AGE": age,
    "EDUCATION_LEVEL": education_level,
    "MARITAL_STATUS": marital_status,
    "NO_OF_KIDS": no_of_kids,
    "OCCUPATION_CATEGORY": occupation_category,
    "INCOME": income,
    "RISK_LEVEL": risk_level,
    "SPENDING_VS_INCOME": spending_vs_income,
    "SPENDING_LEVEL": spending_level,
    "NETWORTH": net_worth
        
}


    # **Convert dictionary to a DataFrame and TRANSPOSE it**
    features = pd.DataFrame(user_data, index=[0])
    return features 
    
df = user_input_features()

# **Display the DataFrame**
st.write("User Input Data:")
st.write(df)

# Load the trained model
model_path = "/Users/sylviabhoke/Downloads/personal_repos folder/Investor_Risk_Tolerance_and_Robo-Advisor/models/best_model.pkl"
model = joblib.load(model_path)

# 🔹 Step 1: Log Transform `INCOME` & `NETWORTH`
columns_to_log_transform = ["INCOME", "NETWORTH"]

# 🔹 Step 2: Encode Categorical Features
df = encode_categorical_features(df)

# 🔹 Step 4: Make Prediction
st.write("### Predicted Risk Tolerance:")
prediction = model.predict(df)
st.write(prediction)
        
        
# Portfolio Optimization
st.write("#### Select your preferred portfolio stock/assets for optimization.")
selected_stocks = st.multiselect("Select Stocks", stock_data.columns)

if selected_stocks:
    sample_data = stock_data[selected_stocks].copy()
    risk_tolerance = prediction[0]
    
    clean_weights, cumulative_returns = optimize_portfolio(sample_data, risk_tolerance)
    
    st.write("#### Portfolio Allocation:")
    st.bar_chart(clean_weights)
    
    # Compute the portfolio value assuming an initial investment of $100
    initial_investment = 100
    portfolio_value = initial_investment * cumulative_returns
    portfolio_value.index = portfolio_value.index.astype(str).str[:4]

    # Display Portfolio Value Growth
    st.write("#### Portfolio Value of $100 Investment Over Time:")
    st.line_chart(portfolio_value)
    






