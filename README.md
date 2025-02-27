# 📈 Investor Risk Tolerance and Robo-Advisor


##  Overview  
This project leverages **machine learning** and **quantitative finance** techniques to help investors:  

✅ Determine their **risk tolerance** through a **questionnaire-based ML model**.  
✅ Optimize their **portfolio allocation** based on their risk level.  
✅ Visualize **portfolio allocation** and **investment performance over time**.  

The app is built using **Streamlit** for a user-friendly interface and integrates **Efficient Frontier optimization** from `PyPortfolioOpt` to suggest optimal asset allocations.  

---

## 🔹 Features  

### 1️⃣ **Risk Tolerance Prediction**  
- Users **fill out a questionnaire** that includes **age, income, spending habits, net worth, and other factors**.  
- A **trained ML model** predicts their **risk tolerance level** based on historical data.  

### 2️⃣ **Portfolio Optimization (Markowitz Model)**  
- Users **select their preferred stocks/assets** from a given list.  
- The app **calculates expected returns and risk metrics**.  
- **Portfolio allocation is optimized** based on risk tolerance using the **Efficient Frontier** method:  
  - **Conservative Investors** → Minimize portfolio volatility 📉  
  - **Aggressive Investors** → Maximize Sharpe Ratio for optimal returns 📈  

### 3️⃣ **Investment Performance Visualization**  
- **📊 Bar Chart**: Displays the **optimized portfolio weights**.  
- **📈 Line Chart**: Shows the **growth of a $100 investment** over time.  

---

  
### Project structure 
```
📁 data/                      # Stores raw and processed datasets  
 ├── 📁 raw/                  # Original unprocessed data  
 │    ├── Risk_data.csv       # Investor responses for risk prediction  
 │    ├── Stock_data.csv      # Market data for portfolio optimization  
 │    ├── Variable_definition.txt  # Dataset variable descriptions  
 │  
 ├── 📁 processed/             # Cleaned and feature-engineered datasets  
 │    ├── processed_data.pkl  # Preprocessed dataset for ML models  

📁 src/                        # Main source code  
 ├── 📁 feature_preprocessing/  # Feature engineering & transformations  
 │    ├── data_cleaning.py     # Data cleaning & preprocessing  
 │    ├── transformations.py   # Log transforms, encoding, scaling  

 ├── 📁 modeling/               # Machine learning models  
 │    ├── train.py             # Training ML models for risk prediction  
 │    ├── predict.py           # Making predictions using trained models  

 ├── 📁 portfolio_optimization/  # Portfolio optimization logic  
 │    ├── allocation.py        # Portfolio allocation using risk scores  
 │    ├── markowitz.py         # Markowitz optimization functions  

📁 models/                     # Saved trained models  
 ├── best_model.pkl            # Best performing risk tolerance model  
 ├── portfolio_optimizer.pkl   # Portfolio optimization model  

📁 notebooks/                   # Jupyter notebooks for EDA & experiments  
 ├── risk_tolerance.ipynb      # ML model training for risk prediction  
 ├── portfolio_optimization.ipynb  # Portfolio optimization experiments  

```
---

### License
This project is open-source and available under the MIT License.

### Contributing
Feel free to open issues, submit pull requests, or suggest improvements. Contributions are always welcome!

### Future Enhancements
* ✅ Expand risk tolerance model with deep learning.
* ✅ Allow users to manually adjust risk levels.
* ✅ Add real-time stock data API integration.

---