import streamlit as st
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix
from pathlib import Path

# page config
st.set_page_config(page_title="Naive Bayes Classification", layout="centered")


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
        <h1>Naive Bayes Classification</h1>
        <p>Predict <b>Customer Gender</b> using <b>Billing Details</b></p>
        <p>Algorithm Used: <b>Gaussian Naive Bayes</b></p>
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

# select features
X = df[["total_bill", "tip", "size"]]
y = df["sex"]  # classification target

# train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# model
model = GaussianNB()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# metrics
accuracy = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

# performance
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("Model Performance")

c1, c2 = st.columns(2)
c1.metric("Accuracy", f"{accuracy * 100:.2f}%")
c2.metric("Samples Tested", len(y_test))

st.markdown("</div>", unsafe_allow_html=True)

# confusion matrix
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("Confusion Matrix")

fig, ax = plt.subplots()
ax.imshow(cm)
ax.set_xticks([0, 1])
ax.set_yticks([0, 1])
ax.set_xticklabels(model.classes_)
ax.set_yticklabels(model.classes_)
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")

for i in range(2):
    for j in range(2):
        ax.text(j, i, cm[i, j], ha="center", va="center", color="white")

st.pyplot(fig)
st.markdown("</div>", unsafe_allow_html=True)

# prediction
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("Predict Customer Gender")

bill = st.slider(
    "Total Bill ($)", float(df["total_bill"].min()), float(df["total_bill"].max())
)
tip = st.slider("Tip ($)", float(df["tip"].min()), float(df["tip"].max()))
size = st.slider("Table Size", int(df["size"].min()), int(df["size"].max()))

input_data = scaler.transform([[bill, tip, size]])
prediction = model.predict(input_data)[0]
probability = model.predict_proba(input_data).max()

st.markdown(
    f"""
    <div class="prediction-box">
        Predicted Gender: <b>{prediction}</b><br>
        Confidence: {probability * 100:.2f}%
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown("</div>", unsafe_allow_html=True)
