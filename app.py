
import streamlit as st
import pandas as pd
import joblib
import os

# Page settings
st.set_page_config(
    page_title="Sales Prediction App",
    page_icon="📊",
    layout="centered"
)

# Load saved model
MODEL_PATH = "sales_prediction_model.pkl"

st.title("📊 Sales Prediction App")
st.write("Predict product sales using advertising spending.")

if not os.path.exists(MODEL_PATH):
    st.error("Model file not found. Please run python app.py first to train and save the model.")
    st.stop()

model = joblib.load(MODEL_PATH)

st.subheader("Enter Advertising Budget")

# User inputs
tv = st.number_input(
    "TV Advertising Budget",
    min_value=0.0,
    max_value=1000.0,
    value=200.0,
    step=10.0
)

radio = st.number_input(
    "Radio Advertising Budget",
    min_value=0.0,
    max_value=1000.0,
    value=30.0,
    step=5.0
)

newspaper = st.number_input(
    "Newspaper Advertising Budget",
    min_value=0.0,
    max_value=1000.0,
    value=20.0,
    step=5.0
)

# Predict button
if st.button("Predict Sales"):
    input_data = pd.DataFrame(
        [[tv, radio, newspaper]],
        columns=["TV", "Radio", "Newspaper"]
    )

    prediction = model.predict(input_data)[0]

    st.success(f"Predicted Sales: {prediction:.2f}")

    st.info(
        "This is an estimate from your trained machine learning model, "
        "not a guarantee of actual sales."
    )

st.divider()
st.caption("Model: Linear Regression | Dataset: Advertising.csv")