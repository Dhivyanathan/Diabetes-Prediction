import pandas as pd
import joblib
import streamlit as st
import warnings

from warnings import filterwarnings
filterwarnings('ignore')

# -----------------------------
# Load Trained Models
# -----------------------------
def load_models():
    model = joblib.load("model_final.pkl")
    gender_encoder = joblib.load("gender_encoder_final.pkl")
    smoking_encoder = joblib.load("smoking_encoder_final.pkl")
    return model, gender_encoder, smoking_encoder

# -----------------------------
# Calling Functions
# -----------------------------
model, gender_encoder, smoking_encoder = load_models()

# -----------------------------
# Prediction Function
# -----------------------------
def give_prediction(
    gender,
    age,
    hypertension,
    heart_disease,
    smoking_history,
    bmi,
    hba1c,
    glucose,
    model,
    gender_encoder,
    smoking_encoder
):
    # Encode Categorical Inputs
    gender_encoded = gender_encoder.transform([gender])[0]
    smoking_encoded = smoking_encoder.transform([smoking_history])[0]

    # Build Feature Row
    input_data = pd.DataFrame([[
        gender_encoded,
        age,
        hypertension,
        heart_disease,
        smoking_encoded,
        bmi,
        hba1c,
        glucose
    ]], columns=[
        "gender",
        "age",
        "hypertension",
        "heart_disease",
        "smoking_history",
        "bmi",
        "HbA1c_level",
        "blood_glucose_level"
    ])

    # Return Prediction
    return model.predict(input_data)[0]

# -----------------------------
# Streamlit Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Diabetes Prediction System",
    layout="centered"
)

# -----------------------------
# Title
# -----------------------------
st.title("🩺 Diabetes Prediction System")

# -----------------------------
# Description
# -----------------------------
st.write(
    "Check your diabetes risk based on basic health details"
)

# -----------------------------
# Input Fields
# -----------------------------
gender = st.selectbox(
    "Gender :",
    ["Female", "Male", "Other"]
)

age = st.number_input(
    "Age :",
    min_value=0.0,
    max_value=100.0
)

hypertension_choice = st.selectbox(
    "Hypertension :",
    ["No", "Yes"]
)

heart_disease_choice = st.selectbox(
    "Heart Disease :",
    ["No", "Yes"]
)

smoking_history = st.selectbox(
    "Smoking History :",
    ["No Info", "never", "former", "current", "not current", "ever"]
)

bmi = st.number_input(
    "BMI :",
    min_value=0.0
)

hba1c = st.number_input(
    "HbA1c Level :",
    min_value=0.0
)

glucose = st.number_input(
    "Blood Glucose Level :",
    min_value=0
)

# -----------------------------
# Encode Yes / No Choices
# -----------------------------
hypertension = 1 if hypertension_choice == "Yes" else 0
heart_disease = 1 if heart_disease_choice == "Yes" else 0

# -----------------------------
# Prediction Button
# -----------------------------
if st.button("Predict"):
    result = give_prediction(
        gender,
        age,
        hypertension,
        heart_disease,
        smoking_history,
        bmi,
        hba1c,
        glucose,
        model,
        gender_encoder,
        smoking_encoder
    )

    st.subheader("Prediction Result")

    if result == 1:
        st.error("The person is likely Diabetic")
    else:
        st.success("The person is likely Non-Diabetic")

# -----------------------------
# Footer
# -----------------------------
st.markdown("---")
st.markdown(
    "This app uses a Random Forest Classifier"
)
