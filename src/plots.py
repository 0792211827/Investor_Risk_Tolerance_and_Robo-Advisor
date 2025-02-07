import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_pickle('/Users/sylviabhoke/Downloads/personal_repos folder/Investor_Risk_Tolerance_and_Robo-Advisor/data/interim/processed_data.pkl')

df



def different_features(df):
    """
    Identifies discrete and continuous features in the DataFrame.
    """
    discrete_features = [col for col in df.columns if len(df[col].unique()) < 25 and col not in ['RISK07']]
    continuous_features = [col for col in df.columns if col not in discrete_features and col not in ['RISK07']]
    
    return discrete_features, continuous_features

discrete_features, continuous_features = different_features(df)

discrete_features, continuous_features

def plot_features(df, discrete_features, continuous_features):
    """
    Plots count plots for discrete features and distribution plots for continuous features.
    """
    # Plot count plots for discrete features
    for col in discrete_features:
        plt.figure(figsize=(8, 4))
        sns.countplot(data=df, x=col, palette="viridis")
        plt.title(f'Count Plot of {col}')
        plt.xticks(rotation=45)
        plt.show()
    
    # Plot distribution plots for continuous features
    for col in continuous_features:
        plt.figure(figsize=(8, 4))
        sns.histplot(df[col], kde=True)
        plt.title(f'Distribution Plot of {col}')
        plt.show()

discrete_features, continuous_features = different_features(df)
plot_features(df, discrete_features, continuous_features)