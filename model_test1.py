# test_model.py
import pandas as pd
import joblib

# --- Step 1: Load trained model ---
model = joblib.load("house_price_model.pkl")

# --- Step 2: Define test houses ---
# Each row: [area_sqft, bedrooms, bathrooms, swimming_pool,
#            parking_space, furnished, furniture_available,
#            locality_score, access_main_road, shopping_complex, local_market]

test_houses = pd.DataFrame(
    [
        [2200, 3, 2, 1, 2, 1, 1, 4, 1, 1, 1],  # mid-range house with amenities
        [1500, 2, 1, 0, 1, 0, 0, 3, 0, 0, 1],  # smaller, unfurnished, basic locality
        [3500, 5, 3, 1, 3, 1, 1, 5, 1, 1, 1],  # luxury house with all features
    ],
    columns=[
        "area_sqft", "bedrooms", "bathrooms", "swimming_pool",
        "parking_space", "furnished", "furniture_available",
        "locality_score", "access_main_road", "shopping_complex", "local_market"
    ]
)

# --- Step 3: Predict prices ---
predictions = model.predict(test_houses)

# --- Step 4: Display results ---
for i, price in enumerate(predictions):
    print(f"House {i+1} predicted price: ₹{price:,.0f}")
