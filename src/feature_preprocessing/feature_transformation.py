import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder



df = pd.read_pickle('/Users/sylviabhoke/Downloads/personal_repos folder/Investor_Risk_Tolerance_and_Robo-Advisor/data/interim/processed_data.pkl')

df 

def log_transformation(df, columns_to_log_transform):
    """
    Plots the distribution of data before and after log transformation.
    """
    for col in columns_to_log_transform:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Before transformation
        sns.histplot(df[col], kde=True, ax=axes[0])
        axes[0].set_title(f'Before Log Transformation: {col}')
        
        # Apply log transformation
        df[col] = np.log1p(df[col])
        
        # After transformation
        sns.histplot(df[col], kde=True, ax=axes[1])
        axes[1].set_title(f'After Log Transformation: {col}')
        
        plt.show()
    
    return df

df = log_transformation(df, ['INCOME', 'NETWORTH'])


def encode_categorical_features(df):
    """
    Encodes categorical features using Label Encoding.
    """
    le = LabelEncoder()
    df['EDUCATION_LEVEL'] = le.fit_transform(df['EDUCATION_LEVEL'])
    df['MARITAL_STATUS'] = le.fit_transform(df['MARITAL_STATUS'])
    df['OCCUPATION_CATEGORY'] = le.fit_transform(df['OCCUPATION_CATEGORY'])
    df['SPENDING_VS_INCOME'] = le.fit_transform(df['SPENDING_LEVEL'])
    return df

df = encode_categorical_features(df)


def standard_scale_features(df):
    """
    Performs standard scaling on all features except the target column 'RISK_TOLERANCE'.
    """
    features_to_scale = [col for col in df.columns if col != 'RISK_TOLERANCE']
    scaler = StandardScaler()
    df[features_to_scale] = scaler.fit_transform(df[features_to_scale])
    return df

df = standard_scale_features(df)




df.to_pickle('/Users/sylviabhoke/Downloads/personal_repos folder/Investor_Risk_Tolerance_and_Robo-Advisor/data/processed/processed_data.pkl')



