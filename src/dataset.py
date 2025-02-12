import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


df = pd.read_csv('/Users/sylviabhoke/Downloads/personal_repos folder/Investor_Risk_Tolerance_and_Robo-Advisor/data/raw/Risk_data.csv')

df

def data_sanity(df):
    """
    Perform basic data sanity checks on a Pandas DataFrame.
    Disp
    lays key dataset information and statistics.
    """
    print("\nFirst 5 rows of the DataFrame:")
    print(df.head())
    
    print("\nDataFrame Info:")
    print(df.info())
    
    print("\nColumn Names:")
    print(df.columns)
    
    print("\nNumber of Duplicated Rows:", df.duplicated().sum())
    
    print("\nStatistical Summary:")
    print(df.describe())
    
    print("\nMissing Values per Column:")
    print(df.isnull().sum())
    
    # dropping some columns
    columns_to_drop = ['Unnamed: 0', 'LIFECL07']
    for col in columns_to_drop:
       if col in df.columns:
          df = df.drop(col, axis=1)
    print(f"\nDropped '{col}' column")
  
    return df

def rename_columns(df):
    """
    Renames specific columns in the DataFrame.
    """
    rename_dict = {
        'AGE07': 'AGE', 'EDCL07': 'EDUCATION_LEVEL', 'MARRIED07': 'MARITAL_STATUS',
        'KIDS07': 'NO_OF_KIDS', 'OCCAT107': 'OCCUPATION_CATEGORY', 'INCOME07': 'INCOME',
        'RISK07': 'RISK_LEVEL', 'WSAVED07': 'SPENDING_VS_INCOME', 'SPENDMOR07': 'SPENDING_LEVEL',
        'NETWORTH07': 'NETWORTH', 'TrueRiskTol': 'RISK_TOLERANCE'
    }
    
    df = df.rename(columns=rename_dict)
    return df

df = data_sanity(df)

df = rename_columns(df)

print("\nRenamed Columns:", df.columns)





df.to_pickle('/Users/sylviabhoke/Downloads/personal_repos folder/Investor_Risk_Tolerance_and_Robo-Advisor/data/interim/processed_data.pkl')
