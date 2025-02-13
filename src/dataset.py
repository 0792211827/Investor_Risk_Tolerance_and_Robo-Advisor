import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf


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


def fetch_stock_data(stock_tickers, start_date, end_date):
    """
    Fetches historical stock data from Yahoo Finance and extracts the closing prices.

    Parameters:
    - stock_tickers (list): List of stock symbols.
    - start_date (str): Start date in 'YYYY-MM-DD' format.
    - end_date (str): End date in 'YYYY-MM-DD' format.

    Returns:
    - pd.DataFrame: DataFrame with dates as index and stock tickers as columns (closing prices).
    """
    # Download the stock data
    stock_data = yf.download(stock_tickers, start=start_date, end=end_date)

    # Extract only columns containing 'Close' prices
    stock_data = stock_data[[col for col in stock_data.columns if 'Close' in col]]

    # Select only 'Close' prices and drop the 'Price' level
    data = stock_data.xs('Close', level='Price', axis=1)

    # Remove the 'Ticker' name from columns
    data.columns.name = None

    return data

# Example usage
start_date = '2020-01-01'
end_date = '2025-01-31'
stock_tickers = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META", "NVDA", "NFLX", "BRK-B", "JPM",
    "V", "MA", "PYPL", "DIS", "ADBE", "INTC", "AMD", "IBM", "CSCO", "ORCL",
    "PFE", "MRNA", "JNJ", "UNH", "BABA", "KO", "PEP", "MCD", "NKE", "T",
    "XOM", "CVX", "BP", "BA", "CAT", "GE", "GS", "SPGI", "LMT", "MMM",
    "TSM", "SHOP", "SQ", "UBER", "LYFT", "ZM", "TWLO", "CRWD", "DOCU", "WMT"
]

data = fetch_stock_data(stock_tickers, start_date, end_date)
data.head()


data.to_csv('/Users/sylviabhoke/Downloads/personal_repos folder/Investor_Risk_Tolerance_and_Robo-Advisor/data/raw/stock_data.csv',reset_index=True)