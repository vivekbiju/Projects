import streamlit as st
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --- PAGE CONFIG ---
st.set_page_config(page_title="House Price Insights", page_icon="🏡", layout="wide")


@st.cache_resource
def load_assets():
    model = joblib.load('house_model.pkl')
    model_features = model.get_booster().feature_names
    return model, model_features


model, model_features = load_assets()

st.title("🏡 House Price Predictor & Insights")

# Use columns for inputs
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Property Inputs")
    overall_qual = st.slider("Overall Quality (1-10)", 1, 10, 7)
    gr_liv_area = st.number_input("Current Living Area (sqft)", 500, 5000, 1800)
    year_built = st.slider("Year Built", 1880, 2026, 2005)
    garage_cars = st.selectbox("Garage Cars", [0, 1, 2, 3], index=2)
    total_bsmt_sf = st.number_input("Basement (sqft)", 0, 3000, 1000)


# --- PREDICTION LOGIC ---
def get_prediction(area_input):
    df = pd.DataFrame(0.0, index=[0], columns=model_features)
    # Map inputs
    inputs = {
        'OverallQual': overall_qual,
        'GrLivArea': area_input,
        'YearBuilt': year_built,
        'GarageCars': garage_cars,
        'TotalBsmtSF': total_bsmt_sf
    }
    for key, val in inputs.items():
        if key in df.columns:
            df[key] = float(val)

    pred = model.predict(df)[0]
    return np.expm1(pred) if pred < 50 else pred


# Calculate current price
current_price = get_prediction(gr_liv_area)

with col2:
    st.subheader("Price Prediction")
    st.metric(label="Estimated Value", value=f"${current_price:,.2f}")

    # --- VISUALIZATION SECTION ---
    st.subheader("Price Trend by Square Footage")

    # Generate data for the graph (Area from 500 to 5000)
    area_range = np.linspace(500, 5000, 20)
    prices = [get_prediction(a) for a in area_range]

    # Plotting with Matplotlib
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(area_range, prices, color='#ff4b4b', linewidth=2, marker='o', markersize=4)

    # Highlight the current selection
    ax.scatter(gr_liv_area, current_price, color='black', s=100, zorder=5, label='Your Selection')

    ax.set_xlabel("Living Area (sqft)")
    ax.set_ylabel("Price ($)")
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend()

    # Display in Streamlit
    st.pyplot(fig)

st.divider()
st.info("💡 Notice how the red dot moves along the line as you change the 'Current Living Area' input!")