import pandas as pd
import joblib
import numpy as np

# --- Step 1: Load trained models ---
reg_model = joblib.load("house_price_model.pkl")
clf_model = joblib.load("house_trend_model.pkl")

# --- Step 2: Generate 100 synthetic test properties ---
np.random.seed(42)

property_types = ["Villa", "House", "Flat", "Empty Land", "Building", "Farm House", "Apartment"]
residence_statuses = ["Residence", "Non-Residence"]
registration_statuses = ["Registrable", "Documented"]

test_properties = []

for i in range(100):
    p_type = np.random.choice(property_types)
    res_status = np.random.choice(residence_statuses)
    reg_status = np.random.choice(registration_statuses)

    # Floor logic
    if p_type == "House":
        floor = np.random.randint(1, 4)
    elif p_type == "Building":
        floor = np.random.randint(1, 21)
    elif p_type in ["Flat", "Apartment"]:
        floor = np.random.randint(1, 31)
    else:
        floor = 0

    test_properties.append([
        np.random.randint(500, 75000),   # area_sqft
        np.random.randint(0, 10),        # bedrooms
        np.random.randint(0, 6),         # bathrooms
        np.random.choice([0, 1]),        # swimming_pool
        np.random.randint(0, 6),         # parking_space
        np.random.choice([0, 1]),        # furnished
        np.random.choice([0, 1]),        # furniture_available
        np.random.randint(1, 11),        # locality_score
        np.random.choice([0, 1]),        # access_main_road
        np.random.choice([0, 1]),        # shopping_complex
        np.random.choice([0, 1]),        # local_market
        np.random.randint(100000, 5000000), # previous_price
        floor,                           # floor
        p_type,                          # property_type
        res_status,                      # residence_status
        reg_status                       # registration_status
    ])

columns = [
    "area_sqft", "bedrooms", "bathrooms", "swimming_pool",
    "parking_space", "furnished", "furniture_available",
    "locality_score", "access_main_road", "shopping_complex",
    "local_market", "previous_price", "floor",
    "property_type", "residence_status", "registration_status"
]

test_df = pd.DataFrame(test_properties, columns=columns)

# --- Step 3: Predict prices and trends ---
predicted_prices = reg_model.predict(test_df)
predicted_trends = clf_model.predict(test_df)

# --- Step 4: Display results ---
for i in range(len(test_df)):
    print(f"Property {i+1}:")
    print(f"  🏠 Predicted Price: ₹{predicted_prices[i]:,.0f}")
    print(f"  📊 Future Trend: {predicted_trends[i]}")
    print(f"  🏢 Type: {test_df.iloc[i]['property_type']}, Floor: {test_df.iloc[i]['floor']}")
    print(f"  🏷️ Status: {test_df.iloc[i]['residence_status']} / {test_df.iloc[i]['registration_status']}")
    print("-" * 50)
