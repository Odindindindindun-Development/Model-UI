import streamlit as st
import numpy as np
import pandas as pd
import joblib

# Load model and scaler
model = joblib.load("best_model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("🩺 Diabetes Prediction App")
st.write("Enter your health metrics to predict the likelihood of diabetes.")

# Load CSV to get feature names
df = pd.read_csv("diabetes_binary_5050split_health_indicators_BRFSS2015.csv")
X = df.drop('Diabetes_binary', axis=1)
feature_names = X.columns

st.subheader("🔢 Input Features")
inputs = []
for feature in feature_names:
    value = st.number_input(feature, value=float(df[feature].mean()))
    inputs.append(value)

input_array = np.array(inputs).reshape(1, -1)
scaled_input = scaler.transform(input_array)

if st.button("Predict Diabetes"):
    pred = model.predict(scaled_input)[0]
    prob = None
    if hasattr(model, "predict_proba"):
        prob = model.predict_proba(scaled_input)[0][1]

    st.write("### 🧠 Result:")
    if pred == 1:
        st.error(f"⚠️ Diabetes Detected! Probability: {prob:.2f}" if prob is not None else "⚠️ Diabetes Detected!")
    else:
        st.success(f"✅ No Diabetes Detected. Probability: {prob:.2f}" if prob is not None else "✅ No Diabetes Detected.")
