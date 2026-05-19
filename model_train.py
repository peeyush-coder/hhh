import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib

# --- Step 1: Load dataset ---
df = pd.read_csv("house_data_with_floors.csv")

# --- Step 2: Define features ---
numeric_features = [
    "area_sqft", "bedrooms", "bathrooms", "swimming_pool",
    "parking_space", "furnished", "furniture_available",
    "locality_score", "access_main_road", "shopping_complex",
    "local_market", "previous_price", "floor"
]

categorical_features = [
    "property_type", "residence_status", "registration_status"
]

X = df[numeric_features + categorical_features]
y_price = df["price"]                # regression target
y_trend = df["future_trend"]         # classification target (Up/Down)

# --- Step 3: Preprocessing ---
preprocessor = ColumnTransformer(
    transformers=[
        ("num", "passthrough", numeric_features),
        ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), categorical_features)
    ]
)

# --- Step 4: Split data ---
X_train, X_test, y_price_train, y_price_test = train_test_split(
    X, y_price, test_size=0.2, random_state=42
)
_, _, y_trend_train, y_trend_test = train_test_split(
    X, y_trend, test_size=0.2, random_state=42
)

# --- Step 5: Regression pipeline ---
reg_model = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", LinearRegression())
])
reg_model.fit(X_train, y_price_train)

# --- Step 6: Evaluate regression model ---
y_price_pred = reg_model.predict(X_test)
mse = mean_squared_error(y_price_test, y_price_pred)
r2 = r2_score(y_price_test, y_price_pred)

print("🏠 Regression Model (Price Prediction)")
print("Mean Squared Error:", mse)
print("R² Score:", r2)

# --- Step 7: Classification pipeline ---
clf_model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(max_iter=2000))
])
clf_model.fit(X_train, y_trend_train)

# --- Step 8: Evaluate classification model ---
y_trend_pred = clf_model.predict(X_test)
accuracy = accuracy_score(y_trend_test, y_trend_pred)

print("\n📈 Classification Model (Future Trend Prediction)")
print("Accuracy:", accuracy)

# --- Step 9: Save models ---
joblib.dump(reg_model, "house_price_model.pkl")
joblib.dump(clf_model, "house_trend_model.pkl")
print("\nModels saved as house_price_model.pkl and house_trend_model.pkl")
