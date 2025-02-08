import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler



df = pd.read_pickle('/Users/sylviabhoke/Downloads/personal_repos folder/Investor_Risk_Tolerance_and_Robo-Advisor/data/interim/processed_data.pkl')

df 

def plot_log_transformation(df, columns_to_log_transform):
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

df = plot_log_transformation(df, ['INCOME', 'NETWORTH'])


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



