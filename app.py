import streamlit as st
import pandas as pd
import joblib
import numpy as np
import sys
import os
from sklearn.preprocessing import LabelEncoder
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import risk_models, expected_returns

# Import custom modules
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(project_root)
from src.modeling.predict import make_predictions
from src.feature_preprocessing.feature_transformation import log_transformation, encode_categorical_features
from src.portfolio_optimization.markowitz import calculate_portfolio_statistics, optimize_portfolio

# Load stock data
stock_data = pd.read_csv('/Users/sylviabhoke/Downloads/personal_repos folder/Investor_Risk_Tolerance_and_Robo-Advisor/data/raw/stock_data.csv', index_col='Date')

# ---- PAGE TITLE & DESCRIPTION ----
st.title("📊 Investor Risk Tolerance & Robo-Advisor")

st.markdown("""
    This app is a simple robo-advisor that uses a machine learning model to predict the risk tolerance of an investor based on their responses to a questionnaire. The app then uses the predicted risk tolerance to optimize a portfolio allocation using the Markowitz Portfolio Optimization model.
    
    **📌 Steps to Follow:**
    - **Step 1:** Enter your investor characteristics to predict your risk tolerance.
    - **Step 2:** Select your preferred stocks, and the app will generate the optimal portfolio.
""")

# ---- SIDEBAR: INVESTOR QUESTIONNAIRE ----
st.sidebar.header("📝 Step 1: Enter Investor Characteristics")

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

# Display user inputs
st.markdown("### 📋 User Input Summary")
st.dataframe(df)

# ---- RISK TOLERANCE PREDICTION ----
st.markdown("## 📈 Step 2: Asset Allocation and Portfolio Performance")

# Load trained model
model_path = "/Users/sylviabhoke/Downloads/personal_repos folder/Investor_Risk_Tolerance_and_Robo-Advisor/models/best_model.pkl"
model = joblib.load(model_path)

# Transform input data
df = encode_categorical_features(df)

# Predict risk tolerance
predicted_risk_tolerance = model.predict(df)[0]
st.write(f"**Predicted Risk Tolerance Score:** {predicted_risk_tolerance:.2f} (scale of 100)")

# ---- PORTFOLIO OPTIMIZATION ----
st.write("### 📊 Portfolio Optimization")
selected_stocks = st.multiselect("Select Stocks", stock_data.columns)

if st.button("SUBMIT"):
    if selected_stocks:
        sample_data = stock_data[selected_stocks].copy()
        clean_weights, cumulative_returns = optimize_portfolio(sample_data, predicted_risk_tolerance)
        
        st.markdown("#### 📌 Asset Allocation - Mean-Variance Allocation")
        st.bar_chart(clean_weights)

        # Compute portfolio value assuming an initial investment of $100
        initial_investment = 100
        portfolio_value = initial_investment * cumulative_returns


        st.markdown("#### 📈 Portfolio Value of $100 Investment Over Time")
        st.line_chart(portfolio_value)





        
        




    






