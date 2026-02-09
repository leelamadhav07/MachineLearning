import streamlit as st
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from pathlib import Path

# page config
st.set_page_config(page_title="KNN Regression", layout="centered")


# load css
def load_css(file):
    css_file = Path(__file__).parent / file
    with open(css_file) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


load_css("style.css")

# title
st.markdown(
    """
    <div class="card">
        <h1>KNN Regression</h1>
        <p>Upload your dataset and let's predict the future!</p>
        <p> Predict <b> Tip Amount </b> from <b> Total Bill </b> using KNN Regression</p>
    </div>
    """,
    unsafe_allow_html=True,
)


# load data
@st.cache_data
def load_data():
    return sns.load_dataset("tips")


df = load_data()

# dataset preview
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("Dataset Preview")
st.dataframe(df.head())
st.markdown("</div>", unsafe_allow_html=True)

# prepare data
X = df[["total_bill"]]
y = df["tip"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# K selection
st.markdown('<div class="card">', unsafe_allow_html=True)
k = st.slider("Select Number of Neighbors (K)", 1, 20, 5)
st.markdown("</div>", unsafe_allow_html=True)

# train model
model = KNeighborsRegressor(n_neighbors=k)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)
adj_r2 = 1 - (1 - r2) * (len(y_test) - 1) / (len(y_test) - 2)

# visualization
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("Total Bill Vs Tip (KNN Prediction)")

# sorted values for smooth curve
X_sorted = np.sort(X.values, axis=0)
X_sorted_scaled = scaler.transform(X_sorted)
y_sorted_pred = model.predict(X_sorted_scaled)

fig, ax = plt.subplots()
ax.scatter(df["total_bill"], df["tip"], alpha=0.6)
ax.plot(X_sorted, y_sorted_pred, color="red")
ax.set_xlabel("Total Bill")
ax.set_ylabel("Tip")
st.pyplot(fig)

st.markdown("</div>", unsafe_allow_html=True)

# performance
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("Model Performance")

c1, c2 = st.columns(2)
c1.metric("MAE", f"{mae:.2f}")
c2.metric("RMSE", f"{rmse:.2f}")

c3, c4 = st.columns(2)
c3.metric("R²", f"{r2:.3f}")
c4.metric("Adj R²", f"{adj_r2:.3f}")

st.markdown("</div>", unsafe_allow_html=True)

# prediction
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("Predict Tip Amount")

bill = st.slider(
    "Total Bill $", float(df["total_bill"].min()), float(df["total_bill"].max())
)

bill_scaled = scaler.transform([[bill]])
tip = model.predict(bill_scaled)[0]

st.markdown(
    f'<div class="prediction-box">Predict Tip : ${tip:.2f}</div>',
    unsafe_allow_html=True,
)

st.markdown("</div>", unsafe_allow_html=True)
