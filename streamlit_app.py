import streamlit as st
import pandas as pd
import joblib

# -----------------------------
# Load trained model and features
# -----------------------------
model = joblib.load("fraud_model_resampled.pkl")
features = joblib.load("model_resampled_features.pkl")

# -----------------------------
# Streamlit Page Setup
# -----------------------------
st.set_page_config(
    page_title="NovaPay Fraud Detection",
    page_icon="💳",
    layout="centered"
)

st.title("💳 NovaPay Fraud Detection System")
st.write("Enter transaction details to check if the transaction is fraudulent.")

st.divider()

# -----------------------------
# User Inputs
# -----------------------------
amount = st.number_input(
    "Transaction Amount",
    min_value=0.0,
    value=100.0
)

hour = st.slider(
    "Transaction Hour",
    0,
    23,
    12
)

customer_txn_count = st.number_input(
    "Customer Transaction Count",
    min_value=0,
    value=1
)

avg_customer_amount = st.number_input(
    "Average Customer Amount",
    min_value=0.0,
    value=50.0
)

merchant_risk = st.slider(
    "Merchant Risk Score",
    0.0,
    1.0,
    0.1
)

st.divider()

# -----------------------------
# Prepare Input Data
# -----------------------------
input_data = pd.DataFrame({
    'amount': [amount],
    'hour': [hour],
    'customer_txn_count': [customer_txn_count],
    'avg_customer_amount': [avg_customer_amount],
    'merchant_risk': [merchant_risk]
})

# Align with training features (fixes fragmentation warning)
input_data = input_data.reindex(columns=features, fill_value=0)

# -----------------------------
# Prediction
# -----------------------------
if st.button("🔍 Predict Fraud"):

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    st.subheader("Prediction Result")

    if prediction == 1:
        st.error("⚠ Fraudulent Transaction Detected")
    else:
        st.success("✅ Legitimate Transaction")

    st.metric(
        label="Fraud Probability",
        value=f"{probability:.2%}"
    )

    st.divider()

    st.write("### Input Summary")

    st.write({
        "Amount": amount,
        "Transaction Hour": hour,
        "Customer Transaction Count": customer_txn_count,
        "Average Customer Amount": avg_customer_amount,
        "Merchant Risk": merchant_risk
    })
