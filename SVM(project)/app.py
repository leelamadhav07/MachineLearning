import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
import os
import requests
import pandas as pd
from datetime import datetime
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler, LabelEncoder

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="END to END SVM", layout="centered", initial_sidebar_state="collapsed"
)


# ---------------- LOGGING ----------------
def log(message):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(timestamp, message)


# ---------------- SESSION STATE ----------------
if "cleaned_saved" not in st.session_state:
    st.session_state.cleaned_saved = False

if "df_clean" not in st.session_state:
    st.session_state.df_clean = None

# ---------------- FOLDER SETUP ----------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
CLEAN_DIR = os.path.join(BASE_DIR, "data", "cleaned")

os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(CLEAN_DIR, exist_ok=True)

log("Application started")
log(f"RAW_DIR = {RAW_DIR}")
log(f"CLEAN_DIR = {CLEAN_DIR}")

# ---------------- UI HEADER ----------------
st.title("SVM Platform")

# ---------------- SIDEBAR ----------------
st.sidebar.header("SVM Settings")

kernel = st.sidebar.selectbox("Kernel", ["linear", "rbf", "poly", "sigmoid"])

C = st.sidebar.slider("C (Regularisation)", 0.01, 10.0, 1.0)

gamma = st.sidebar.selectbox("Gamma", ["scale", "auto"])

log(f"Settings → kernel={kernel}, C={C}, gamma={gamma}")

# ---------------- STEP 1 : DATA INGESTION ----------------
st.header("Step 1: Data Ingestion")
log("Data ingestion started")

option = st.radio(
    "Choose Data Source", ["select", "Download dataset", "Upload dataset"]
)

df = None
raw_path = None

if option == "Download dataset":
    if st.button("Download Iris Dataset"):
        log("Downloading iris dataset")

        url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv"
        response = requests.get(url)
        response.raise_for_status()

        raw_path = os.path.join(RAW_DIR, "iris.csv")
        with open(raw_path, "wb") as f:
            f.write(response.content)

        df = pd.read_csv(raw_path)

        st.success("Dataset downloaded successfully")
        log(f"Iris dataset saved at {raw_path}")

elif option == "Upload dataset":
    upload_file = st.file_uploader("Upload CSV", type=["csv"])

    if upload_file is not None:
        raw_path = os.path.join(RAW_DIR, upload_file.name)

        with open(raw_path, "wb") as f:
            f.write(upload_file.getbuffer())

        df = pd.read_csv(raw_path)

        st.success("File uploaded successfully")
        log(f"Uploaded at {raw_path}")

# ---------------- STEP 2 : EDA ----------------
if df is not None:
    st.header("Step 2: Exploratory Data Analysis")
    log("EDA started")

    st.dataframe(df.head())
    st.write("Shape:", df.shape)
    st.write("Missing values:", df.isna().sum())

    numeric_df = df.select_dtypes(include=np.number)

    if numeric_df.shape[1] > 1:
        fig, ax = plt.subplots()
        sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", ax=ax)
        st.pyplot(fig)

    log("EDA completed")
else:
    st.info("Please complete Step 1 first")

# ---------------- STEP 3 : DATA CLEANING ----------------
if df is not None:
    st.header("Step 3: Data Cleaning")

    strategy = st.selectbox("Missing value strategy", ["Mean", "Median", "Drop rows"])

    df_clean = df.copy()

    if strategy == "Drop rows":
        df_clean = df_clean.dropna()
    else:
        for col in df_clean.select_dtypes(include=np.number):
            if strategy == "Mean":
                df_clean[col] = df_clean[col].fillna(df_clean[col].mean())
            else:
                df_clean[col] = df_clean[col].fillna(df_clean[col].median())

    st.session_state.df_clean = df_clean
    st.success("Data cleaning completed")
    log("Data cleaning completed")

# ---------------- STEP 4 : SAVE CLEANED DATA ----------------
if st.button("Save cleaned Dataset", disabled=st.session_state.df_clean is None):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    clean_filename = f"cleaned_dataset_{timestamp}.csv"
    clean_path = os.path.join(CLEAN_DIR, clean_filename)

    st.session_state.df_clean.to_csv(clean_path, index=False)

    st.success("Cleaned dataset saved")
    st.info(f"Saved at: {clean_path}")
    log(f"Saved cleaned dataset to {clean_path}")

# ---------------- STEP 5 : LOAD CLEANED DATA ----------------
st.header("Step 5: Load Cleaned Dataset")

clean_files = [f for f in os.listdir(CLEAN_DIR) if f.endswith(".csv")]

if not clean_files:
    st.warning("No cleaned datasets found. Please complete Step 4.")
else:
    selected = st.selectbox("Select cleaned dataset", clean_files)

    df_model = pd.read_csv(os.path.join(CLEAN_DIR, selected))

    st.success("Dataset loaded successfully")
    log(f"Loaded cleaned dataset: {selected}")

    st.dataframe(df_model.head())

# ---------------- STEP 6 : TRAIN SVM ----------------
st.header("Step 6: Train SVM")
log("Training started")

if "df_model" not in locals():
    st.error("Please load a cleaned dataset first")
    st.stop()

target = st.selectbox("Select target column", "species")

y = df_model[target]
if y.dtype == "object":
    y = LabelEncoder().fit_transform(y)
    log("Target column encoded")

x = df_model.drop(columns=[target])
x = x.select_dtypes(include=np.number)

if x.empty:
    st.error("No numeric features available")
    st.stop()

# Scaling
scaler = StandardScaler()
x = scaler.fit_transform(x)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    x, y, test_size=0.25, random_state=42
)

# Model
model = SVC(kernel=kernel, gamma=gamma, C=C)
model.fit(X_train, y_train)

# Evaluation
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

st.success(f"Accuracy: {acc:.2f}")
log(f"SVM trained successfully | Accuracy = {acc:.2f}")

cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt="d", cmap="coolwarm", ax=ax)
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
st.pyplot(fig)
