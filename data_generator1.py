# data_generator.py
import pandas as pd
import numpy as np

# --- Step 1: Set number of samples ---
num_samples = 500
np.random.seed(42)  # reproducibility

# --- Step 2: Generate synthetic features ---
area_sqft = np.random.randint(500, 4000, num_samples)        # house size
bedrooms = np.random.randint(1, 6, num_samples)              # 1–5 bedrooms
bathrooms = np.random.randint(1, 4, num_samples)             # 1–3 bathrooms
swimming_pool = np.random.choice([0, 1], num_samples)        # 0 = No, 1 = Yes
parking_space = np.random.randint(0, 4, num_samples)         # 0–3 vehicles
furnished = np.random.choice([0, 1], num_samples)            # 0 = No, 1 = Yes
furniture_available = np.random.choice([0, 1], num_samples)  # 0 = No, 1 = Yes
locality_score = np.random.randint(1, 6, num_samples)        # 1–5 rating
access_main_road = np.random.choice([0, 1], num_samples)     # 0 = No, 1 = Yes
shopping_complex = np.random.choice([0, 1], num_samples)     # 0 = No, 1 = Yes
local_market = np.random.choice([0, 1], num_samples)         # 0 = No, 1 = Yes

# --- Step 3: Generate synthetic target (price) ---
price = (
    50000
    + area_sqft * 60
    + bedrooms * 25000
    + bathrooms * 20000
    + swimming_pool * 50000
    + parking_space * 15000
    + furnished * 30000
    + furniture_available * 20000
    + locality_score * 10000
    + access_main_road * 25000
    + shopping_complex * 20000
    + local_market * 15000
    + np.random.randint(-30000, 30000, num_samples)  # noise
)

# --- Step 4: Create DataFrame ---
df = pd.DataFrame({
    "area_sqft": area_sqft,
    "bedrooms": bedrooms,
    "bathrooms": bathrooms,
    "swimming_pool": swimming_pool,
    "parking_space": parking_space,
    "furnished": furnished,
    "furniture_available": furniture_available,
    "locality_score": locality_score,
    "access_main_road": access_main_road,
    "shopping_complex": shopping_complex,
    "local_market": local_market,
    "price": price
})

# --- Step 5: Save to CSV ---
df.to_csv("house_data.csv", index=False)
print("Synthetic dataset saved as house_data.csv with", num_samples, "rows.")
