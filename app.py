import streamlit as st
import numpy as np
import pandas as pd
import joblib

# Load model and scaler
model = joblib.load("best_model.pkl")
scaler = joblib.load("scaler.pkl")

# App title
st.set_page_config(page_title="Diabetes Prediction App", page_icon="🩺", layout="centered")
st.title("🩺 Diabetes Prediction App")
st.markdown("### Enter your health metrics below to predict your likelihood of diabetes.")

# Load dataset for feature names
df = pd.read_csv("diabetes_binary_5050split_health_indicators_BRFSS2015.csv")
X = df.drop('Diabetes_binary', axis=1)
feature_names = X.columns

# Create three columns layout
st.markdown("## 🧩 Input Features")
col1, col2, col3 = st.columns(3)

inputs = []
for i, feature in enumerate(feature_names):
    with [col1, col2, col3][i % 3]:
        # Use sliders when feature range makes sense
        mean_val = float(df[feature].mean())
        min_val = float(df[feature].min())
        max_val = float(df[feature].max())

        # If feature likely binary (0 or 1), use selectbox
        if set(df[feature].unique()) == {0, 1}:
            value = st.selectbox(f"{feature}", [0, 1], index=int(mean_val))
        else:
            value = st.slider(f"{feature}", min_val, max_val, mean_val)
        inputs.append(value)

# Scale input
input_array = np.array(inputs).reshape(1, -1)
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
