import streamlit as st
import joblib
import pandas as pd

# --- Load models ---
reg_model = joblib.load("house_price_model.pkl")
clf_model = joblib.load("house_trend_model.pkl")

# --- Page config ---
st.set_page_config(page_title="🏠 Property Price Estimator", layout="wide")

# --- Title ---
st.title("🏠 Property Price Estimation")
st.markdown("Enter property details below to estimate the price and see future trend.")

# --- Input form (sidebar) ---
with st.sidebar.form("property_form"):
    st.header("🏡 Enter Property Details")

    # Numeric inputs
    area_sqft = st.number_input("Area (sqft)", 500, 75000, 2200, step=100)
    bedrooms = st.slider("Bedrooms", 0, 10, 3)
    bathrooms = st.slider("Bathrooms", 0, 6, 2)
    swimming_pool = st.checkbox("Swimming Pool")
    parking_space = st.slider("Parking Spaces", 0, 6, 2)
    furnished = st.checkbox("Furnished")
    furniture_available = st.checkbox("Furniture Available")
    locality_score = st.slider("Locality Score (1-10)", 1, 10, 7)
    access_main_road = st.checkbox("Access to Main Road")
    shopping_complex = st.checkbox("Nearby Shopping Complex")
    local_market = st.checkbox("Nearby Local Market")
    previous_price = st.number_input("Previous Price (₹)", 100000, 5000000, 500000, step=50000)

    # New categorical inputs
    property_type = st.selectbox("Property Type", 
        ["Villa", "House", "Flat", "Empty Land", "Building", "Farm House", "Apartment"])
    residence_status = st.selectbox("Residence Status", ["Residence", "Non-Residence"])
    registration_status = st.selectbox("Registration Status", ["Registrable", "Documented"])

    # Floor input (only relevant for House, Building, Flat, Apartment)
    floor = 0
    if property_type in ["House"]:
        floor = st.slider("Floors (House)", 1, 3, 2)
    elif property_type in ["Building"]:
        floor = st.slider("Floors (Building)", 1, 20, 5)
    elif property_type in ["Flat", "Apartment"]:
        floor = st.slider("Floor (Flat/Apartment)", 1, 30, 2)

    submitted = st.form_submit_button("Estimate Price")

# --- After estimation ---
if submitted:
    # Prepare input
    new_property = pd.DataFrame(
        [[area_sqft, bedrooms, bathrooms, int(swimming_pool),
          parking_space, int(furnished), int(furniture_available),
          locality_score, int(access_main_road), int(shopping_complex),
          int(local_market), previous_price, floor,
          property_type, residence_status, registration_status]],
        columns=[
            "area_sqft", "bedrooms", "bathrooms", "swimming_pool",
            "parking_space", "furnished", "furniture_available",
            "locality_score", "access_main_road", "shopping_complex",
            "local_market", "previous_price", "floor",
            "property_type", "residence_status", "registration_status"
        ]
    )

    # Predictions
    predicted_price = reg_model.predict(new_property)[0]
    predicted_trend = clf_model.predict(new_property)[0]

    # Results
    st.success(f"💰 Estimated Price: ₹{predicted_price:,.0f}")
    if predicted_trend == "Up":
        st.markdown("<span style='color:green; font-size:20px;'>⬆️ Price Trend: UP</span>", unsafe_allow_html=True)
    else:
        st.markdown("<span style='color:red; font-size:20px;'>⬇️ Price Trend: DOWN</span>", unsafe_allow_html=True)

    # Show summary table
    st.markdown("### 🏡 Property Summary")
    st.write(new_property)
