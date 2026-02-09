import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.tree import DecisionTreeRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

# ---------------- LOAD DATA ----------------
df = pd.read_csv("housing.csv")

# ---------------- HANDLE MISSING VALUES ----------------
imputer = SimpleImputer(strategy="median")
df["total_bedrooms"] = imputer.fit_transform(df[["total_bedrooms"]])

# ---------------- FEATURES & TARGET ----------------
X = df.drop("median_house_value", axis=1)
y = df["median_house_value"]

# ---------------- TRAIN TEST SPLIT ----------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ---------------- PREPROCESSING ----------------
cat_features = ["ocean_proximity"]
num_features = X.columns.drop(cat_features)

preprocessor = ColumnTransformer(
    [
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_features),
        ("num", "passthrough", num_features),
    ]
)

X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

# ---------------- MODEL ----------------
model = DecisionTreeRegressor(max_depth=10, min_samples_split=4, random_state=42)

model.fit(X_train_processed, y_train)

# ---------------- EVALUATION ----------------
y_pred = model.predict(X_test_processed)

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

# ---------------- STREAMLIT UI ----------------
st.title("🏠 California Housing Price Prediction (Regression)")

st.subheader("Model Performance")
st.write("MAE:", round(mae, 2))
st.write("RMSE:", round(rmse, 2))
st.write("R² Score:", round(r2, 3))

st.subheader("Dataset Preview")
st.dataframe(df.head())

st.subheader("Enter House Details")

longitude = st.number_input("Longitude", value=-122.23)
latitude = st.number_input("Latitude", value=37.88)
housing_median_age = st.number_input("Housing Median Age", value=30)
total_rooms = st.number_input("Total Rooms", value=1000)
total_bedrooms = st.number_input("Total Bedrooms", value=200)
population = st.number_input("Population", value=800)
households = st.number_input("Households", value=300)
median_income = st.number_input("Median Income", value=4.0)

ocean_proximity = st.selectbox(
    "Ocean Proximity",
    ["NEAR BAY", "<1H OCEAN", "INLAND", "NEAR OCEAN", "ISLAND"],
)

# ---------------- PREDICTION ----------------
if st.button("Predict House Price"):
    input_df = pd.DataFrame(
        [
            {
                "longitude": longitude,
                "latitude": latitude,
                "housing_median_age": housing_median_age,
                "total_rooms": total_rooms,
                "total_bedrooms": total_bedrooms,
                "population": population,
                "households": households,
                "median_income": median_income,
                "ocean_proximity": ocean_proximity,
            }
        ]
    )

    input_processed = preprocessor.transform(input_df)
    prediction = model.predict(input_processed)[0]

    st.success(f"Predicted Median House Value: ${prediction:,.2f}")
