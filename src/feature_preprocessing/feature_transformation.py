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
    Applies log transformation to specified columns.
    """
    for col in columns_to_log_transform:
        df[col] = np.log1p(df[col])  

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



df.to_pickle('/Users/sylviabhoke/Downloads/personal_repos folder/Investor_Risk_Tolerance_and_Robo-Advisor/data/processed/processed_data.pkl')



