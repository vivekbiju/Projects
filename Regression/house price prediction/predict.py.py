import joblib
import pandas as pd
import xgboost
import numpy as np
import os


def predict():
    # --- STEP 0: FIX PATHS ---
    # This ensures the script finds the files even if you run it from a different folder
    base_path = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_path, 'house_model.pkl')
    columns_path = os.path.join(base_path, 'model_columns.pkl')

    # --- STEP 1: LOAD MODEL ---
    try:
        model = joblib.load(model_path)
    except FileNotFoundError:
        print(f"ERROR: Could not find {model_path}. Check the file location!")
        return

    # --- STEP 2: GET FEATURES ---
    try:
        # Try to get features directly from the XGBoost model
        if hasattr(model, 'feature_names_in_'):
            model_features = model.feature_names_in_
        else:
            model_features = model.get_booster().feature_names

        if model_features is None:
            raise ValueError("Booster has no feature names")
    except:
        # Fallback to the columns file
        try:
            model_features = joblib.load(columns_path)
        except FileNotFoundError:
            print("ERROR: model_columns.pkl not found. Cannot align features.")
            return

    # --- STEP 3: DATA ---
    input_data = {
        'OverallQual': 7,
        'GrLivArea': 1800,
        'GarageCars': 2,
        'TotalBsmtSF': 1000,
        'FullBath': 2,
        'YearBuilt': 2005
    }

    # --- STEP 4 & 5: ALIGNMENT ---
    # Create DataFrame with all 0s for the exact columns the model expects
    df = pd.DataFrame(0, index=[0], columns=model_features)

    # Fill in the data we have
    for key, value in input_data.items():
        if key in df.columns:
            df[key] = value

    # --- STEP 6: PREDICT WITH ERROR HANDLING ---
    try:
        prediction = model.predict(df.values)

        # Handle log transformation if necessary
        final_price = prediction[0]
        if final_price < 20:
            final_price = np.expm1(final_price)

        print("\n" + "=" * 35)
        print("  HOUSE PRICE PREDICTION SUCCESS  ")
        print("=" * 35)
        print(f"Estimated Market Value: ${final_price:,.2f}")
        print("=" * 35)

    except ValueError as e:
        print("\n" + "!" * 35)
        print("VALUE ERROR ENCOUNTERED")
        print(f"Details: {e}")
        print("-" * 35)
        print(f"Model expects: {len(model_features)} features")
        print(f"You provided: {df.shape[1]} features")
        print("!" * 35)


if __name__ == "__main__":
    predict()