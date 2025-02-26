# 📈 Investor Risk Tolerance and Robo-Advisor


## Overview
This project leverages machine learning and quantitative finance techniques to help investors:
✅ Determine their risk tolerance through a questionnaire-based ML model.
✅ Optimize their portfolio allocation based on their risk level.
✅ Visualize portfolio allocation and investment performance over time.

The app is built using Streamlit for a user-friendly interface and integrates Efficient Frontier optimization from PyPortfolioOpt to suggest optimal asset allocations.

## Features
**1️ Risk Tolerance Prediction**
* Users fill out a questionnaire that includes age, income, spending habits, net worth, and other factors.
* A trained ML model predicts their risk tolerance level based on historical data.
**2️ Portfolio Optimization (Markowitz Model)**
* Users select their preferred stocks/assets from a given list.
* The app calculates expected returns and risk metrics.
* Portfolio allocation is optimized based on risk tolerance using the Efficient Frontier method:
    * Conservative Investors → Minimize portfolio volatility 📉
    * Aggressive Investors → Maximize Sharpe Ratio for optimal returns 📈
**3️ Investment Performance Visualization**
* Bar Chart: Displays the optimized portfolio weights.
* Line Chart: Shows the growth of a $100 investment over time.


### Project structure 

Investor_Risk_Tolerance_and_Robo-Advisor/
│── app.py                      # Main Streamlit app  
│── requirements.txt             # Dependencies  
│── README.md                    # Project documentation  
│  
├── 📁 data/                      # Stores raw, interim, and processed data  
│    ├── 📁 raw/                  # Unprocessed datasets (original CSV files)  
│    │    ├── Risk_data.csv       # Investor data for risk tolerance modeling  
│    │    ├── Stock_data.csv      # Market data for portfolio optimization  
│    │    ├── Variable_definition.txt  # Descriptions of dataset variables  
│    │  
│    ├── 📁 processed/             # Cleaned and feature-engineered datasets  
│    │    ├── processed_data.pkl  # Preprocessed dataset for ML models  
│    │  
│    ├── 📁 external/              # Any externally sourced datasets  
│    ├── 📁 interim/               # Temporary storage during transformations  
│  
├── 📁 src/                        # Main source code  
│    ├── 📜 __init__.py            # Marks src as a package  
│    ├── 📜 config.py              # Global configuration settings  
│    ├── 📜 dataset.py             # Functions for loading and processing datasets  
│    │  
│    ├── 📁 feature_preprocessing/  # Feature engineering & preprocessing  
│    │    ├── 📜 __init__.py  
│    │    ├── 📜 feature_transformations.py # Log transforms, standard scaling, etc.  
│    │  
│    ├── 📁 modeling/               # Machine Learning models  
│    │    ├── 📜 __init__.py  
│    │    ├── 📜 train.py           # Training ML models for risk tolerance  
│    │    ├── 📜 predict.py         # Making predictions (used by Streamlit)  
│    │  
│    ├── 📁 portfolio_optimization/  # Markowitz optimization & asset allocation  
│    │    ├── 📜 __init__.py   
│    │    ├── 📜 markowitz.py       # Markowitz optimization functions  
│    │   
│    │  
│    ├── 📁 visualization/          # Data visualization scripts  
│    │    ├── 📜 __init__.py  
│    │    ├── 📜 plots.py           # Functions to generate plots  
│  
├── 📁 models/                     # Saved trained models  
│    ├── best_model.pkl            # Best performing risk tolerance model    
│  
├── 📁 notebooks/                   # Jupyter notebooks for EDA & experiments  
│    ├── risk_tolerance.ipynb       # Model training for risk tolerance  
│    ├── portfolio_optimization.ipynb  # Portfolio optimization experiments  



### License
This project is open-source and available under the MIT License.

### Contributing
Feel free to open issues, submit pull requests, or suggest improvements. Contributions are always welcome!

### Future Enhancements
✅ Expand risk tolerance model with deep learning.
✅ Allow users to manually adjust risk levels.
✅ Add real-time stock data API integration.