import streamlit as st
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="House Price Predictor",
    page_icon="🏡",
    layout="wide"
)


# --- MODEL LOADING (Cloud-Safe Version) ---
@st.cache_resource
def load_assets():
    # This finds the folder where THIS app.py file lives
    base_path = os.path.dirname(__file__)
    model_path = os.path.join(base_path, 'house_model.pkl')

    # Load the trained model
    model = joblib.load(model_path)
    # Extract the exact feature list from the model's booster
    model_features = model.get_booster().feature_names
    return model, model_features


try:
    model, model_features = load_assets()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.info("Check if 'house_model.pkl' is in the same GitHub folder as 'app.py'")
    st.stop()

# --- UI LAYOUT ---
st.title("🏡 House Price Predictor & Insights")
st.markdown("Adjust the property details to see how the market value changes.")

col1, col2 = st.columns([1, 1.2])

with col1:
    st.subheader("Property Details")
    overall_qual = st.slider("Overall Quality (1-10)", 1, 10, 7)
    gr_liv_area = st.number_input("Living Area (sqft)", 500, 5000, 1800)
    year_built = st.slider("Year Built", 1880, 2026, 2005)
    garage_cars = st.selectbox("Garage Car Capacity", [0, 1, 2, 3, 4], index=2)
    total_bsmt_sf = st.number_input("Total Basement (sqft)", 0, 4000, 1000)

    neighborhoods = ['CollgCr', 'Veenker', 'Crawfor', 'NoRidge', 'Mitchel', 'Somerst', 'NridgHt', 'OldTown', 'Edwards',
                     'Gilbert']
    selected_neighborhood = st.selectbox("Neighborhood", sorted(neighborhoods))


# --- PREDICTION HELPER ---
def get_prediction(area_input):
    # Create template of 276 zeros
    df = pd.DataFrame(0.0, index=[0], columns=model_features)

    # Fill in user inputs
    inputs = {
        'OverallQual': overall_qual,
        'GrLivArea': area_input,
        'YearBuilt': year_built,
        'GarageCars': garage_cars,
        'TotalBsmtSF': total_bsmt_sf,
        f"Neighborhood_{selected_neighborhood}": 1.0
    }

    for key, val in inputs.items():
        if key in df.columns:
            df[key] = float(val)

    pred = model.predict(df)[0]
    # Reverse log-scaling if price is a small log-value (e.g. 12.1)
    return np.expm1(pred) if pred < 50 else pred


# Calculate Current Selection
current_price = get_prediction(gr_liv_area)

with col2:
    st.subheader("Prediction Result")
    st.metric(label="Estimated Market Value", value=f"${current_price:,.2f}")

    # --- VISUALIZATION ---
    st.subheader("Price Sensitivity: Living Area")

    # Calculate prices for a range of sizes
    area_range = np.linspace(500, 5000, 25)
    prices = [get_prediction(a) for a in area_range]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(area_range, prices, color='#FF4B4B', linewidth=2.5, label="Price Trend")
    ax.scatter(gr_liv_area, current_price, color='black', s=120, zorder=5, label="Your Selection")

    ax.set_xlabel("Living Area (sqft)")
    ax.set_ylabel("Price ($)")
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend()

    st.pyplot(fig)

st.divider()
st.caption("Machine Learning Model: XGBoost Regression | Dataset: Ames Housing")