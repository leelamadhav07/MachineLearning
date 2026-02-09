import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.tree import DecisionTreeClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report

# Load dataset
df = pd.read_csv("housing.csv")

# Handle missing values
imputer = SimpleImputer(strategy="median")
df["total_bedrooms"] = imputer.fit_transform(df[["total_bedrooms"]])

# Convert target to classes
df["price_class"] = pd.cut(
    df["median_house_value"], bins=[0, 150000, 300000, 500000], labels=[0, 1, 2]
)
df = df.dropna(subset=["price_class"])
# Features and target
X = df.drop(["median_house_value", "price_class"], axis=1)
y = df["price_class"]

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Preprocessing
cat_features = ["ocean_proximity"]
num_features = X.columns.drop(cat_features)

preprocessor = ColumnTransformer(
    [("cat", OneHotEncoder(), cat_features), ("num", "passthrough", num_features)]
)

X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

# Model
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train_processed, y_train)

# Accuracy
y_pred = model.predict(X_test_processed)
acc = accuracy_score(y_test, y_pred)

# ---------------- Streamlit UI ----------------

st.title("🏠 California Housing Price Classifier")

st.write("Model Accuracy:", acc)
st.write("Dataset:", df.head())
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
    "Ocean Proximity", ["NEAR BAY", "<1H OCEAN", "INLAND", "NEAR OCEAN", "ISLAND"]
)

if st.button("Predict Price Class"):
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
    pred_class = model.predict(input_processed)[0]

    label_map = {0: "Low", 1: "Medium", 2: "High"}
    st.success(f"Predicted Price Category: {label_map[int(pred_class)]}")
