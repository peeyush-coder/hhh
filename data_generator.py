import pandas as pd
import numpy as np

# --- Step 1: Set number of samples ---
num_samples = 2500
np.random.seed(42)

# --- Step 2: Generate synthetic features ---
area_sqft = np.random.randint(500, 750000, num_samples)
bedrooms = np.random.randint(1, 15, num_samples)
bathrooms = np.random.randint(1, 10, num_samples)
swimming_pool = np.random.choice([0, 1], num_samples)
parking_space = np.random.randint(0, 10, num_samples)
furnished = np.random.choice([0, 1], num_samples)
furniture_available = np.random.choice([0, 1], num_samples)
locality_score = np.random.randint(1, 11, num_samples)
access_main_road = np.random.choice([0, 1], num_samples)
shopping_complex = np.random.choice([0, 1], num_samples)
local_market = np.random.choice([0, 1], num_samples)

# --- Step 3: Property type categories ---
property_types = [
    "Villa", "House", "Flat", "Empty Land",
    "Building", "Farm House", "Apartment"
]
property_type = np.random.choice(property_types, num_samples)

# Residence / Non-residence
residence_status = np.random.choice(["Residence", "Non-Residence"], num_samples)

# Registrable vs Documented
registration_status = np.random.choice(["Registrable", "Documented"], num_samples)

# --- Step 4: Floor option logic ---
floor = []
for p in property_type:
    if p in ["House"]:
        floor.append(np.random.randint(1, 4))   # 1–3 floors
    elif p in ["Building"]:
        floor.append(np.random.randint(1, 20))  # up to 20 floors
    elif p in ["Flat", "Apartment"]:
        floor.append(np.random.randint(1, 30))  # up to 30 floors
    else:
        floor.append(0)  # Not applicable
floor = np.array(floor)

# --- Step 5: Generate synthetic target (current price) ---
price = (
    100000
    + area_sqft * 50
    + bedrooms * 40000
    + bathrooms * 30000
    + swimming_pool * 100000
    + parking_space * 20000
    + furnished * 50000
    + furniture_available * 30000
    + locality_score * 15000
    + access_main_road * 10000
    + shopping_complex * 10000
    + local_market * 15000
    + floor * 10000
    + np.random.randint(-50000, 50000, num_samples)
)

# --- Step 6: Balanced previous price and future trend ---
half = num_samples // 2
previous_price_up = price[:half] - np.random.randint(50000, 150000, half)
future_trend_up = np.array(["Up"] * half)

previous_price_down = price[half:] + np.random.randint(50000, 150000, num_samples - half)
future_trend_down = np.array(["Down"] * (num_samples - half))

previous_price = np.concatenate([previous_price_up, previous_price_down])
future_trend = np.concatenate([future_trend_up, future_trend_down])

# --- Step 7: Create DataFrame ---
df = pd.DataFrame({
    "property_type": property_type,
    "floor": floor,
    "residence_status": residence_status,
    "registration_status": registration_status,
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
    "previous_price": previous_price,
    "price": price,
    "future_trend": future_trend
})

# --- Step 8: Save to CSV ---
df.to_csv("house_data_with_floors.csv", index=False)
print("Synthetic dataset saved as house_data_with_floors.csv with", num_samples, "rows.")
