# 🏠 House Price Predictor — Linear Regression

An interactive Streamlit app that takes a housing dataset from raw data to a working price predictor: EDA, feature selection, a Linear Regression model, evaluation, coefficient analysis and a Ridge/Lasso comparison.

**🔗 Live demo:** _add your Streamlit link here_

![App screenshot](screenshots/app.png)

## ✨ Features
- **EDA:** null check, descriptive statistics, target (price) distribution
- **Feature selection:** correlation bar chart and heatmap, with written reasoning
- **Preprocessing:** median/most-frequent imputation, One-Hot Encoding, scaling
- **Model:** scikit-learn `LinearRegression` on an 80/20 train/test split
- **Evaluation:** MSE, RMSE, MAE and R² (train vs test), actual-vs-predicted plot, residual plot and histogram
- **Coefficient analysis:** which features raise or lower the price most
- **Bonus:** Linear vs Ridge vs Lasso comparison with an adjustable alpha
- **Live predictor:** enter a house's details and get an estimated price

## 🛠️ Tech stack
Python · pandas · NumPy · scikit-learn · Plotly · Streamlit

## 📂 Dataset
The app runs immediately on built-in demo data. To use a real dataset, download **House Prices: Advanced Regression Techniques** (`train.csv`) or the Ames Housing dataset from Kaggle, choose **Upload CSV** in the sidebar and select `SalePrice` as the target.

## 🚀 Run locally
```bash
git clone https://github.com/mdamaanali-ai/FUTURE_DS_01.git
cd FUTURE_DS_01
python -m venv venv
venv\Scripts\activate        # macOS/Linux: source venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## 📁 Project structure
```
├── app.py               # Streamlit application
├── requirements.txt
├── .streamlit/config.toml   # dark theme
└── README.md
```

## 📊 Results
_Run the app on your dataset and add your own numbers here, for example:_

| Model | R² (test) | RMSE |
|---|---|---|
| Linear Regression | – | – |
| Ridge | – | – |
| Lasso | – | – |

## 🔍 Key takeaways
- Area and location are typically the strongest price drivers.
- Residuals should look like a random cloud around 0; patterns suggest non-linearity.
- Ridge/Lasso help when features are highly correlated (multicollinearity).

## 👤 Author
**mdamaanali-ai** · [GitHub](https://github.com/mdamaanali-ai)

Built as part of the Future Interns Data Science internship (Level 2, Task 1).
