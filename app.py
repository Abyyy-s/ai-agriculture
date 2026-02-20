import streamlit as st
import pandas as pd
import joblib

# page configuration
st.set_page_config(
    page_title="AI Agriculture Advisor",
    page_icon="🌱",
    layout="centered"
)

# load model
model = joblib.load("crop_model.pkl")

# ---------- HEADER ----------
st.title("🌱 AI Agriculture Advisor")
st.markdown(
"""
Predict the best crop based on soil nutrients and environmental conditions.
"""
)

st.divider()

# ---------- INPUT SECTION ----------
st.subheader("Enter Soil & Weather Data")

col1, col2 = st.columns(2)

with col1:
    N = st.number_input("Nitrogen (N)", min_value=0)
    P = st.number_input("Phosphorus (P)", min_value=0)
    K = st.number_input("Potassium (K)", min_value=0)
    temperature = st.number_input("Temperature (°C)")

with col2:
    humidity = st.number_input("Humidity (%)")
    ph = st.number_input("pH Value")
    rainfall = st.number_input("Rainfall (mm)")

st.divider()

# ---------- PREDICTION ----------
if st.button("🌾 Predict Best Crop"):

    sample = pd.DataFrame(
        [[N,P,K,temperature,humidity,ph,rainfall]],
        columns=["N","P","K","temperature","humidity","ph","rainfall"]
    )

    prediction = model.predict(sample)

    st.success(f"✅ Recommended Crop: **{prediction[0].upper()}**")

    st.balloons()

# ---------- FOOTER ----------
st.markdown("---")
st.caption("Built using Machine Learning & Streamlit 🚀")
