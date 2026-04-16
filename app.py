import streamlit as st
import numpy as np
import joblib
import pandas as pd

# Load
knn_model = joblib.load("knn_regression.pkl")
gnb_model = joblib.load("gaussian_nb.pkl")
scaler = joblib.load("scaler.pkl")

st.set_page_config(page_title="Student_performance App", layout="wide")

st.title("📊 Student_performance_data")

# Sidebar
model_choice = st.sidebar.selectbox(
    "Select Model",
    ["KNN Regression", "GaussianNB Classification"]
)

# -------------------------------
# ✅ ORIGINAL FEATURE NAMES (IMPORTANT)
# -------------------------------
columns = [
    "Age",
    "Gender",
    "Ethnicity",
    "ParentalEducation",
    "StudyTimeWeekly",
    "Absences",
    "Tutoring",
    "ParentalSupport",
    "Extracurricular",
    "Sports",
    "Music",
    "Volunteering",
    "GPA"
]

# -------------------------------
# Inputs
# -------------------------------
st.subheader("🧾 Enter Input Features")

features = []

for col in columns:
    val = st.number_input(f"{col}", value=0.0)
    features.append(val)

# DataFrame with correct names
input_df = pd.DataFrame([features], columns=columns)

# -------------------------------
# Prediction
# -------------------------------
if st.button("🔍 Predict"):

    try:
        input_scaled = scaler.transform(input_df)

        if model_choice == "KNN Regression":
            prediction = knn_model.predict(input_scaled)
            st.success(f"📈 Result: {prediction[0]:.4f}")
        else:
            prediction = gnb_model.predict(input_scaled)
            st.success(f"🎯 Result: {prediction[0]}")

    except Exception as e:
        st.error(f"❌ Error: {e}")
