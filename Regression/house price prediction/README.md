# 🏡 House Price Predictor

A machine learning web application that predicts house prices using **XGBoost** and **Streamlit**.

## 🚀 Live Demo
[Link to your Streamlit App]

## 📊 Project Overview
This project uses the Ames Housing Dataset to predict market values based on key features like square footage, overall quality, and neighborhood. It features an interactive dashboard with real-time price trends.

## 🛠️ Tech Stack
- **Model:** XGBoost Regressor
- **Frontend:** Streamlit
- **Data:** Pandas, NumPy, Scikit-learn
- **Plots:** Matplotlib

## ⚙️ How to Run Locally
1. Clone the repo: `git clone <your-repo-link>`
2. Install requirements: `pip install -r requirements.txt`
3. Run the app: `streamlit run app.py`


Ames Housing Price Prediction: Advanced Regression Techniques
This project focuses on predicting the sale price of residential homes in Ames, Iowa. Using a dataset with 79 explanatory variables, the goal is to build a robust regression model that can accurately estimate property values through detailed feature engineering and advanced modeling. 
Project Overview
Predicting house prices is a classic "regression" problem. The challenge lies in handling a diverse mix of data—everything from square footage and number of rooms to qualitative measures like "kitchen quality" or "neighborhood prestige."

Key Data Science 
Steps1. Data Cleaning & Outlier RemovalData isn't always perfect. We started by removing massive outliers (e.g., houses with over 4,500 sq. ft. of living area) that didn't follow the typical price-to-size relationship.Code logic: train = train[train['GrLivArea'] < 4500]
2. Target TransformationTo satisfy the statistical assumptions of linear models, we applied a Log Transformation ($log(1+x)$) to the SalePrice.Benefit: This converts a "skewed" distribution into a "normal" bell curve, making the model less sensitive to extreme luxury home prices.
3. Smart Imputation (Filling Missing Values)We didn't just fill every empty box with zero. We used specific strategies:Categorical defaults: Filling "Functional" with "Typical" as per documentation.Grouped Medians: Filling missing "Lot Frontage" (street length) based on the Median of the specific neighborhood.Grouped Modes: Filling "Zoning" based on the house's "SubClass" type.
4. Feature EngineeringBox-Cox Transformation: Automatically detected and fixed "skewed" numerical columns to make them more symmetrical.One-Hot Encoding: Used pd.get_dummies to turn text categories (like "Neighborhood" or "RoofStyle") into numbers that a calculator can understand.
5. Advanced ModelingWe implemented Regularized Regression models to prevent overfitting:Lasso Regression: Automatically performs feature selection by setting the importance of useless variables to zero.Ridge Regression: Helps manage "multicollinearity" (when two features, like Garage Cars and Garage Area, are almost identical).

Results & Evaluation
The models were evaluated using Root Mean Squared Error (RMSE) on the log-transformed price.

Evaluation Function: We used a custom evaluation tool to visualize residuals (the difference between predicted and actual prices) to ensure the model wasn't making biased mistakes.

How to use this project
Environment: Ensure you have pandas, numpy, seaborn, and sklearn installed.

Data: Place the train.csv and test.csv in the project root.

Run: Execute the notebook cells sequentially to process the data, train the models, and generate the submission.csv.

The project generates a final submission file where:

Prices are predicted using the trained Ridge/Lasso model.

Log prices are converted back to dollars using np.expm1.

Data is formatted into the required Id and SalePrice columns for competition entry.
