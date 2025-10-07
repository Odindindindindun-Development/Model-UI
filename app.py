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

# --- Sidebar indicators ---
st.sidebar.markdown("## ℹ️ Feature Value Indicators")
st.sidebar.markdown("""
**Age Group:**  
1 = 18-24, 2 = 25-29, 3 = 30-34, 4 = 35-39, 5 = 40-44, 6 = 45-49, 7 = 50-54, 8 = 55-59, 9 = 60-64, 10 = 65-69, 11 = 70-74, 12 = 75-79, 13 = 80+

**General Health (GenHlth):**  
1 = Excellent, 2 = Very Good, 3 = Good, 4 = Fair, 5 = Poor

**Mental Health (MentHlth):**  
Number of days in the past 30 days that mental health was not good (0–30)

**Physical Health (PhysHlth):**  
Number of days in the past 30 days that physical health was not good (0–30)

**Income:**  
1 = Less than $10,000, 2 = $10,000–$15,000, 3 = $15,000–$20,000, 4 = $20,000–$25,000, 5 = $25,000–$35,000, 6 = $35,000–$50,000, 7 = $50,000–$75,000, 8 = $75,000 or more

**Education:**  
1 = Never attended school or only kindergarten, 2 = Grades 1–8, 3 = Grades 9–11, 4 = Grade 12 or GED, 5 = College 1 year to 3 years, 6 = College 4 years or more

**Binary Features (HighBP, HighChol, Smoker, Stroke, PhysActivity, DiffWalk, CholCheck):**  
No = 0, Yes = 1  
**Sex:** Female = 0, Male = 1

**BMI:**  
Enter your actual Body Mass Index (e.g., 22.5)
""")

# Value mappings
genhlth_map = {1: "Excellent", 2: "Very Good", 3: "Good", 4: "Fair", 5: "Poor"}
income_map = {1: "Less than $10,000", 2: "$10,000–$15,000", 3: "$15,000–$20,000", 4: "$20,000–$25,000", 
              5: "$25,000–$35,000", 6: "$35,000–$50,000", 7: "$50,000–$75,000", 8: "$75,000 or more"}
education_map = {1: "Never attended school or only kindergarten", 2: "Grades 1–8", 3: "Grades 9–11", 
                 4: "Grade 12 or GED", 5: "College 1 year to 3 years", 6: "College 4 years or more"}
age_groups = {1: "18-24", 2: "25-29", 3: "30-34", 4: "35-39", 5: "40-44", 6: "45-49", 7: "50-54", 
              8: "55-59", 9: "60-64", 10: "65-69", 11: "70-74", 12: "75-79", 13: "80+"}

# Load dataset
df = pd.read_csv("diabetes_binary_5050split_health_indicators_BRFSS2015.csv")
X = df.drop('Diabetes_binary', axis=1)
original_features = X.columns

# Display names
display_names = {
    "HighBP": "High Blood Pressure",
    "HighChol": "High Cholesterol",
    "BMI": "Body Mass Index (BMI)",
    "Smoker": "Smoker (Yes/No)",
    "Stroke": "History of Stroke",
    "PhysActivity": "Physical Activity",
    "GenHlth": "General Health (1-5)",
    "DiffWalk": "Difficulty Walking",
    "Age": "Age Group",
    "CholCheck": "Cholesterol Checkup",
    "HvyAlcoholConsump": "Heavy Alcohol Consumption",
    "MentHlth": "Mental Health",
    "Sex": "Sex",
    "PhysHlth": "Physical Health",
    "NoDocbcCost": "No Doctor Because of Cost",
}

# Layout columns
st.markdown("## 🧩 Input Features")
col1, col2, col3 = st.columns(3)
inputs = {}

for i, feature in enumerate(original_features):
    with [col1, col2, col3][i % 3]:
        label = display_names.get(feature, feature)
        mean_val = float(df[feature].mean())
        min_val = float(df[feature].min())
        max_val = float(df[feature].max())

        # Age group
        if feature == "Age":
            age_options = list(age_groups.items())
            default_idx = int(mean_val) - 1 if 1 <= int(mean_val) <= 13 else 0
            selected = st.selectbox(label, options=age_options, format_func=lambda x: f"{x[1]} years", index=default_idx)
            value = selected[0]

        # General Health
        elif feature == "GenHlth":
            options = list(genhlth_map.items())
            default_idx = int(mean_val) - 1 if 1 <= int(mean_val) <= 5 else 0
            selected = st.selectbox(label, options=options, format_func=lambda x: x[1], index=default_idx)
            value = selected[0]

        # Income
        elif feature == "Income":
            options = list(income_map.items())
            default_idx = int(mean_val) - 1 if 1 <= int(mean_val) <= 8 else 0
            selected = st.selectbox(label, options=options, format_func=lambda x: x[1], index=default_idx)
            value = selected[0]

        # Education
        elif feature == "Education":
            options = list(education_map.items())
            default_idx = int(mean_val) - 1 if 1 <= int(mean_val) <= 6 else 0
            selected = st.selectbox(label, options=options, format_func=lambda x: x[1], index=default_idx)
            value = selected[0]

        # Sex
        elif feature == "Sex":
            options = [("Female", 0), ("Male", 1)]
            default_idx = int(mean_val)
            selected = st.selectbox(label, options=options, index=default_idx, format_func=lambda x: x[0])
            value = selected[1]

        # Other binary features
        elif set(df[feature].unique()) == {0, 1}:
            options = [("No", 0), ("Yes", 1)]
            default_idx = int(mean_val)
            selected = st.selectbox(label, options, index=default_idx, format_func=lambda x: x[0])
            value = selected[1]

        # BMI
        elif feature == "BMI":
            value = st.number_input(label, min_value=min_val, max_value=max_val, value=mean_val, step=0.1, format="%.1f")

        # Other numeric features
        else:
            unique_vals = sorted(df[feature].unique())
            value = st.selectbox(label, unique_vals, index=unique_vals.index(mean_val) if mean_val in unique_vals else 0)

        inputs[feature] = value

# Prepare input for model
input_df = pd.DataFrame([inputs], columns=original_features)
scaled_input = scaler.transform(input_df)

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


