import streamlit as st
import numpy as np
import pandas as pd
import joblib

# Load model and scaler
model = joblib.load("best_model.pkl")
scaler = joblib.load("scaler.pkl")

# Page setup
st.set_page_config(page_title="Diabetes Prediction App", page_icon="🩺", layout="centered")
st.title("🩺 Diabetes Prediction App")
st.markdown("### Enter your health metrics below to predict your likelihood of diabetes.")

# Load dataset and feature names
df = pd.read_csv("diabetes_binary_5050split_health_indicators_BRFSS2015.csv")
X = df.drop('Diabetes_binary', axis=1)
original_features = X.columns

# ✅ Display names (for UI only)
display_names = {
    "HighBP": "High Blood Pressure",
    "HighChol": "High Cholesterol",
    "BMI": "Body Mass Index (BMI)",
    "Smoker": "Smoker (Yes/No)",
    "Stroke": "History of Stroke",
    "PhysActivity": "Physical Activity",
    "GenHlth": "General Health (1-5)",
    "DiffWalk": "Difficulty Walking",
    # add more as needed
}

# Create columns for layout
st.markdown("## 🧩 Input Features")
col1, col2, col3 = st.columns(3)

inputs = {}

for i, feature in enumerate(original_features):
    with [col1, col2, col3][i % 3]:
        label = display_names.get(feature, feature)  # Show friendly name if available
        mean_val = float(df[feature].mean())
        min_val = float(df[feature].min())
        max_val = float(df[feature].max())

        # Binary or continuous input
        if set(df[feature].unique()) == {0, 1}:
            value = st.selectbox(label, [0, 1], index=int(mean_val))
        else:
            value = st.slider(label, min_val, max_val, mean_val)
        inputs[feature] = value  # keep original column name for model

# Prepare input for model
input_array = np.array([inputs[f] for f in original_features]).reshape(1, -1)
scaled_input = scaler.transform(input_array)

# Predict button
st.markdown("---")
if st.button("🔍 Predict Diabetes", use_container_width=True):
    pred = model.predict(scaled_input)[0]
    prob = model.predict_proba(scaled_input)[0][1] if hasattr(model, "predict_proba") else None

    st.markdown("## 🧠 Prediction Result")
    if pred == 1:
        st.error(f"⚠️ **Diabetes Detected!**\n\nEstimated Probability: `{prob:.2f}`" if prob else "⚠️ **Diabetes Detected!**")
    else:
        st.success(f"✅ **No Diabetes Detected.**\n\nEstimated Probability: `{prob:.2f}`" if prob else "✅ **No Diabetes Detected.**")

# Footer
st.markdown("---")
st.caption("Built with ❤️ using Streamlit")
