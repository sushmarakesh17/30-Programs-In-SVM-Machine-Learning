import streamlit as st
import pandas as pd
import pickle


# Load trained files
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
encoder = pickle.load(open("encoder.pkl", "rb"))


# Page configuration
st.set_page_config(
    page_title="SVM Student Performance Predictor",
    page_icon="📊",
    layout="centered"
)


# Title
st.title("📊 SVM Student Performance Prediction")
st.write("Predict whether a student will Pass or Fail using Support Vector Machine (SVM).")


# Input fields

study_hours = st.number_input(
    "Study Hours",
    min_value=0,
    max_value=15,
    value=5
)

attendance = st.number_input(
    "Attendance (%)",
    min_value=0,
    max_value=100,
    value=75
)

previous_score = st.number_input(
    "Previous Score",
    min_value=0,
    max_value=100,
    value=60
)

sleep_hours = st.number_input(
    "Sleep Hours",
    min_value=0,
    max_value=12,
    value=7
)

assignments = st.number_input(
    "Assignments Completed",
    min_value=0,
    max_value=10,
    value=6
)


# Prediction button
if st.button("Predict Result"):

    input_data = pd.DataFrame(
        [[
            study_hours,
            attendance,
            previous_score,
            sleep_hours,
            assignments
        ]],
        columns=[
            "Study_Hours",
            "Attendance",
            "Previous_Score",
            "Sleep_Hours",
            "Assignments_Completed"
        ]
    )


    # Scale input
    input_scaled = scaler.transform(input_data)


    # Prediction
    prediction = model.predict(input_scaled)


    # Convert output back to label
    result = encoder.inverse_transform(prediction)


    if result[0] == "Pass":
        st.success(f"Prediction: Student will {result[0]} ✅")
    else:
        st.error(f"Prediction: Student will {result[0]} ❌")


# Dataset preview
st.subheader("Dataset Preview")

df = pd.read_csv("dataset.csv")

st.dataframe(df)