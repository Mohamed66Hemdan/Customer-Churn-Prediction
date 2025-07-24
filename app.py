import streamlit as st
import pandas as pd
import joblib


def binary_transform(X):
    return X.replace({'Yes': 1, 'No': 0, 'Female': 1, 'Male': 0})


st.markdown("""
    <style>
        .block-container {
            max-width: 95%;
            padding-left: 3rem;
            padding-right: 3rem;
            padding-top: 2rem;
        }
    </style>
""", unsafe_allow_html=True)

# Load trained pipeline model
model = joblib.load("svc_pipeline.pkl")

st.title("📊 Customer Churn Prediction App")

# --- Main page inputs in columns ---
st.subheader("👤 Customer Information")

col1, spacer1, col2, spacer2, col3, spacer3, col4 = st.columns([1, 0.3, 1, 0.3, 1, 0.3, 1])

with col1:
    gender = st.selectbox("Gender", ['Female', 'Male'])
    st.markdown("<br>", unsafe_allow_html=True)

    SeniorCitizen = st.selectbox("Senior Citizen", [0, 1])
    st.markdown("<br>", unsafe_allow_html=True)

    Partner = st.selectbox("Partner", ['Yes', 'No'])
    st.markdown("<br>", unsafe_allow_html=True)

    Dependents = st.selectbox("Dependents", ['Yes', 'No'])
    st.markdown("<br>", unsafe_allow_html=True)

    tenure = st.slider("Tenure (Months)", 0, 72, 12)
    st.markdown("<br>", unsafe_allow_html=True)

    

with col2:
    OnlineSecurity = st.selectbox("Online Security", ['Yes', 'No'])
    st.markdown("<br>", unsafe_allow_html=True)

    OnlineBackup = st.selectbox("Online Backup", ['Yes', 'No'])
    st.markdown("<br>", unsafe_allow_html=True)

    DeviceProtection = st.selectbox("Device Protection", ['Yes', 'No'])
    st.markdown("<br>", unsafe_allow_html=True)

    TechSupport = st.selectbox("Tech Support", ['Yes', 'No'])
    st.markdown("<br>", unsafe_allow_html=True)

    MonthlyCharges = st.slider("Monthly Charges", 0.0, 120.0, 70.0)
    st.markdown("<br>", unsafe_allow_html=True)


with col3:
    StreamingTV = st.selectbox("Streaming TV", ['Yes', 'No'])
    st.markdown("<br>", unsafe_allow_html=True)

    StreamingMovies = st.selectbox("Streaming Movies", ['Yes', 'No'])
    st.markdown("<br>", unsafe_allow_html=True)

    MultipleLines = st.selectbox("Multiple Lines", ['Yes', 'No'])
    st.markdown("<br>", unsafe_allow_html=True)

    PhoneService = st.selectbox("Phone Service", ['Yes', 'No'])
    st.markdown("<br>", unsafe_allow_html=True)

    TotalCharges = st.slider("Total Charges", 0.0, 10000.0, 350.0)
    

with col4:
    Contract = st.selectbox("Contract", ['Month-to-month', 'One year', 'Two year'])
    st.markdown("<br>", unsafe_allow_html=True)

    PaperlessBilling = st.selectbox("Paperless Billing", ['Yes', 'No'])
    st.markdown("<br>", unsafe_allow_html=True)

    PaymentMethod = st.selectbox("Payment Method", [
        'Electronic check', 'Mailed check',
        'Bank transfer (automatic)', 'Credit card (automatic)'
    ])
    st.markdown("<br>", unsafe_allow_html=True)

    InternetService = st.selectbox("Internet Service", ['DSL', 'Fiber optic', 'No'])
    st.markdown("<br>", unsafe_allow_html=True)

    


# --- Prepare the input ---
new_data = pd.DataFrame([{
    'gender': gender,
    'SeniorCitizen': SeniorCitizen,
    'Partner': Partner,
    'Dependents': Dependents,
    'tenure': tenure,
    'PhoneService': PhoneService,
    'MultipleLines': MultipleLines,
    'InternetService': InternetService,
    'OnlineSecurity': OnlineSecurity,
    'OnlineBackup': OnlineBackup,
    'DeviceProtection': DeviceProtection,
    'TechSupport': TechSupport,
    'StreamingTV': StreamingTV,
    'StreamingMovies': StreamingMovies,
    'Contract': Contract,
    'PaperlessBilling': PaperlessBilling,
    'PaymentMethod': PaymentMethod,
    'MonthlyCharges': MonthlyCharges,
    'TotalCharges': TotalCharges
}])

# --- Show the input ---
st.markdown("<br>", unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)

st.subheader("📝 Selected Customer Data")
st.write(new_data)

# --- Predict ---
if st.button("🔍 Predict Churn"):
    prediction = model.predict(new_data)[0]
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(new_data)[0][1] * 100
    else:
        proba = None

    if prediction == 1:
        st.error("🔴 The customer is **likely to churn**.")
    else:
        st.success("🟢 The customer is **likely to stay**.")

    if proba is not None:
        st.info(f"Churn probability: **{proba:.2f}%**")
