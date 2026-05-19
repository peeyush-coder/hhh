# model.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import joblib

# --- Step 1: Load dataset ---
df = pd.read_csv("house_data.csv")

# --- Step 2: Define features (X) and target (y) ---
X = df[[
    "area_sqft", "bedrooms", "bathrooms", "swimming_pool",
    "parking_space", "furnished", "furniture_available",
    "locality_score", "access_main_road", "shopping_complex", "local_market"
]]
y = df["price"]

# --- Step 3: Split data ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# --- Step 4: Train model ---
model = LinearRegression()
model.fit(X_train, y_train)

# --- Step 5: Evaluate model ---
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Training complete!")
print("Mean Squared Error:", mse)
print("R² Score:", r2)

# --- Step 6: Save trained model ---
joblib.dump(model, "house_price_model.pkl")
print("Model saved as house_price_model.pkl")

# --- Step 7: Test prediction with new data ---
new_house = pd.DataFrame(
    [[2200, 3, 2, 1, 2, 1, 1, 4, 1, 1, 1]],
    columns=[
        "area_sqft", "bedrooms", "bathrooms", "swimming_pool",
        "parking_space", "furnished", "furniture_available",
        "locality_score", "access_main_road", "shopping_complex", "local_market"
    ]
)

predicted_price = model.predict(new_house)
print("Predicted Price for new house:", predicted_price[0])
