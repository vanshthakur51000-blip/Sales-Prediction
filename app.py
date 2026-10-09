
import streamlit as st
import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Page settings
st.set_page_config(
    page_title="Sales Prediction App",
    page_icon="📊",
    layout="centered"
)

# File paths
MODEL_PATH = "sales_prediction_model.pkl"
DATA_PATH = "Advertising.csv"

# App heading
st.title("📊 Sales Prediction App")
st.write("Predict product sales using advertising spending.")

# Check required files
if not os.path.exists(MODEL_PATH):
    st.error("Model file not found: sales_prediction_model.pkl")
    st.stop()

if not os.path.exists(DATA_PATH):
    st.error("Dataset file not found: Advertising.csv")
    st.stop()

# Load trained model
model = joblib.load(MODEL_PATH)

# Load dataset
df = pd.read_csv(DATA_PATH)

# Remove unnecessary ID column
if "Unnamed: 0" in df.columns:
    df = df.drop(columns=["Unnamed: 0"])

# Advertising budget inputs
st.subheader("Enter Advertising Budget")

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

# Predict sales
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

# ---------------------------------
# SAVE FOUR GRAPHS WITHOUT DISPLAYING
# ---------------------------------

# Graph 1: TV Advertising vs Sales
fig1, ax1 = plt.subplots(figsize=(8, 5))
ax1.scatter(df["TV"], df["Sales"])
ax1.set_xlabel("TV Advertising Budget")
ax1.set_ylabel("Sales")
ax1.set_title("TV Advertising vs Sales")
ax1.grid(True)

fig1.savefig("tv_vs_sales.png", bbox_inches="tight")
plt.close(fig1)

# Graph 2: Radio Advertising vs Sales
fig2, ax2 = plt.subplots(figsize=(8, 5))
ax2.scatter(df["Radio"], df["Sales"])
ax2.set_xlabel("Radio Advertising Budget")
ax2.set_ylabel("Sales")
ax2.set_title("Radio Advertising vs Sales")
ax2.grid(True)

fig2.savefig("radio_vs_sales.png", bbox_inches="tight")
plt.close(fig2)

# Graph 3: Newspaper Advertising vs Sales
fig3, ax3 = plt.subplots(figsize=(8, 5))
ax3.scatter(df["Newspaper"], df["Sales"])
ax3.set_xlabel("Newspaper Advertising Budget")
ax3.set_ylabel("Sales")
ax3.set_title("Newspaper Advertising vs Sales")
ax3.grid(True)

fig3.savefig("newspaper_vs_sales.png", bbox_inches="tight")
plt.close(fig3)

# Graph 4: Actual vs Predicted Sales
X = df[["TV", "Radio", "Newspaper"]]
y = df["Sales"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Train a separate model for the evaluation graph
evaluation_model = LinearRegression()
evaluation_model.fit(X_train, y_train)

y_pred = evaluation_model.predict(X_test)

fig4, ax4 = plt.subplots(figsize=(8, 5))
ax4.scatter(y_test, y_pred, alpha=0.8)

# Ideal prediction reference line
min_value = min(y_test.min(), y_pred.min())
max_value = max(y_test.max(), y_pred.max())

ax4.plot(
    [min_value, max_value],
    [min_value, max_value],
    linestyle="--",
    color="red",
    label="Ideal prediction"
)

ax4.set_xlabel("Actual Sales")
ax4.set_ylabel("Predicted Sales")
ax4.set_title("Actual vs Predicted Sales")
ax4.legend()
ax4.grid(True)

fig4.savefig("actual_vs_predicted_sales.png", bbox_inches="tight")
plt.close(fig4)

# Confirmation in the app
st.divider()
st.caption("Model: Linear Regression | Dataset: Advertising.csv")