import streamlit as st
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="House Price Predictor & Insights",
    page_icon="🏡",
    layout="wide"
)

# --- SMART MODEL LOADING ---
@st.cache_resource
def load_assets():
    # This finds the directory where app.py is located (fixes Cloud pathing issues)
    base_path = os.path.dirname(__file__)
    model_path = os.path.join(base_path, 'house_model.pkl')
    
    # Load the model
    model = joblib.load(model_path)
    # Extract the exact feature list the model expects
    model_features = model.get_booster().feature_names
    return model, model_features

# Load model and features
try:
    model, model_features = load_assets()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# --- HELPER FUNCTION FOR PREDICTION ---
def get_prediction(custom_area):
    # Create a blank template with all 276 columns
    df = pd.DataFrame(0.0, index=[0], columns=model_features)
    
    # Map current UI inputs to the dataframe
    inputs = {
        'OverallQual': overall_qual,
        'GrLivArea': custom_area,
        'YearBuilt': year_built,
        'GarageCars': garage_cars,
        'TotalBsmtSF': total_bsmt_sf,
        'FullBath': full_bath,
        'LotArea': lot_area
    }
    
    # Apply neighborhood encoding
    target_neighborhood_col = f"Neighborhood_{selected_neighborhood}"
    inputs[target_neighborhood_col] = 1

    for key, value in inputs.items():
        if key in df.columns:
            df[key] = float(value)

    # Predict and handle log transformation if necessary
    prediction = model.predict(df)[0]
    return np.expm1(prediction) if prediction < 50 else prediction

# --- UI LAYOUT ---
st.title("🏡 House Price Predictor")
st.markdown("Adjust the details to see how they impact the market value.")

col1, col2 = st.columns([1, 1.2])

with col1:
    st.subheader("Property Details")
    overall_qual = st.slider("Overall Quality (1-10)", 1, 10, 7)
    gr_liv_area = st.number_input("Living Area (sqft)", 500, 5000, 1800)
    year_built = st.slider("Year Built", 1880, 2026, 2005)
    
    neighborhoods = [
        'CollgCr', 'Veenker', 'Crawfor', 'NoRidge', 'Mitchel', 'Somerst', 
        'NridgHt', 'OldTown', 'BrkSide', 'Sawyer', 'SawyerW', 'NAmes', 
        'IDOTRR', 'MeadowV', 'Edwards', 'Timber', 'Gilbert', 'StoneBr', 
        'ClearCr', 'NPkVill', 'Blmngtn', 'BrDale', 'SWISU', 'Blueste'
    ]
    selected_neighborhood = st.selectbox("Neighborhood", sorted(neighborhoods))
    
    garage_cars = st.selectbox("Garage Cars", [0, 1, 2, 3, 4], index=2)
    total_bsmt_sf = st.number_input("Basement (sqft)", 0, 4000, 1000)
    full_bath = st.selectbox("Full Bathrooms", [1, 2, 3, 4], index=1)
    lot_area = st.number_input("Lot Area (sqft)", 500, 50000, 9000)

# Calculate Prediction
current_price = get_prediction(gr_liv_area)

with col2:
    st.subheader("Estimation & Insights")
    st.metric(label="Estimated Market Value", value=f"${current_price:,.2f}")
    
    # --- TREND VISUALIZATION ---
    # Create a range of areas to show the trend
    area_range = np.linspace(500, 5000, 20)
    trend_prices = [get_prediction(a) for a in area_range]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(area_range, trend_prices, color='#FF4B4B', linewidth=2, label="Price Trend")
    ax.scatter(gr_liv_area, current_price, color='black', s=100, zorder=5, label="Current Selection")
    
    ax.set_xlabel("Living Area (sqft)")
    ax.set_ylabel("Price ($)")
    ax.set_title(f"Impact of Space on Price in {selected_neighborhood}")
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.5)
    
    st.pyplot(fig)

st.divider()
st.caption("Data processed using XGBoost Regression and Matplotlib.")
