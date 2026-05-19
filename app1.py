# app.py
import streamlit as st
import joblib
import pandas as pd
import numpy as np

# --- Load trained model ---
model = joblib.load("house_price_model.pkl")

# --- App Title & Intro ---
st.set_page_config(page_title="🏠 House Price Predictor", layout="wide")
st.title("🏠 House Price Prediction")
st.markdown("### Enter house details below to estimate the price")

# --- Sidebar for inputs ---
st.sidebar.header("House Features")

area_sqft = st.sidebar.slider("Area (sqft)", 500, 5000, 2200, step=100)
bedrooms = st.sidebar.slider("Bedrooms", 1, 10, 3)
bathrooms = st.sidebar.slider("Bathrooms", 1, 5, 2)

swimming_pool = st.sidebar.checkbox("Swimming Pool")
parking_space = st.sidebar.slider("Parking Spaces", 0, 5, 2)
furnished = st.sidebar.checkbox("Furnished")
furniture_available = st.sidebar.checkbox("Furniture Available")
locality_score = st.sidebar.slider("Locality Score (1-5)", 1, 5, 4)
access_main_road = st.sidebar.checkbox("Access to Main Road")
shopping_complex = st.sidebar.checkbox("Nearby Shopping Complex")
local_market = st.sidebar.checkbox("Nearby Local Market")

# --- Collect inputs into DataFrame ---
new_house = pd.DataFrame(
    [[
        area_sqft, bedrooms, bathrooms, int(swimming_pool),
        parking_space, int(furnished), int(furniture_available),
        locality_score, int(access_main_road), int(shopping_complex), int(local_market)
    ]],
    columns=[
        "area_sqft", "bedrooms", "bathrooms", "swimming_pool",
        "parking_space", "furnished", "furniture_available",
        "locality_score", "access_main_road", "shopping_complex", "local_market"
    ]
)

# --- Prediction ---
if st.sidebar.button("Predict Price"):
    predicted_price = model.predict(new_house)[0]

    # Show results attractively
    st.success(f"💰 Estimated Price: ₹{predicted_price:,.0f}")

    # Show feature summary
    st.markdown("### 🏡 House Summary")
    st.write(new_house)

    # Visualize prediction
    st.metric("Predicted Price", f"₹{predicted_price:,.0f}")
    st.bar_chart(pd.DataFrame({"Price": [predicted_price]}))
