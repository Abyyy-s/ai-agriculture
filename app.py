import streamlit as st
import pandas as pd
import joblib

# load saved model
model = joblib.load("crop_model.pkl")

st.title("🌱 AI Agriculture Advisor")

st.write("Enter soil and environmental data")

# user inputs
N = st.number_input("Nitrogen (N)")
P = st.number_input("Phosphorus (P)")
K = st.number_input("Potassium (K)")
temperature = st.number_input("Temperature")
humidity = st.number_input("Humidity")
ph = st.number_input("pH Value")
rainfall = st.number_input("Rainfall")

if st.button("Predict Crop"):

    sample = pd.DataFrame(
        [[N,P,K,temperature,humidity,ph,rainfall]],
        columns=["N","P","K","temperature","humidity","ph","rainfall"]
    )

    prediction = model.predict(sample)

    st.success(f"✅ Recommended Crop: {prediction[0]}")
