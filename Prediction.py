import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Loan Prediction", layout="wide")

st.title("🔮 Loan Approval Prediction System")
st.write("Fill applicant details and get instant prediction.")

# Load model
model = joblib.load("models/loan_pipeline.pkl")

data = pd.read_csv("data/LoanApprovalPrediction.csv")
data.drop("Loan_ID", axis=1, inplace=True)

st.sidebar.header("📌 Applicant Information")


def user_input():

    # ======================
    # Categorical Features
    # ======================
    gender = st.sidebar.selectbox("Sex", data["Gender"].unique())
    married = st.sidebar.selectbox("Marital Status", data["Married"].unique())
    education = st.sidebar.selectbox("Education Level", data["Education"].unique())
    self_employed = st.sidebar.selectbox("Employment Type", data["Self_Employed"].unique())

    # ======================
    # 🔥 Family Size (RANGE 0 → 100)
    # ======================
    dependents = st.sidebar.slider(
        "Family Size (Dependents)",
        min_value=0,
        max_value=100,
        value=0,
        step=1,
        help="Number of dependents supported by applicant"
    )

    st.sidebar.markdown("### 💰 Financial Info")

    applicant_income = st.sidebar.slider(
        "Applicant Income",
        min_value=0,
        max_value=50000,
        value=5000,
        step=500
    )

    coapplicant_income = st.sidebar.slider(
        "Coapplicant Income",
        min_value=0,
        max_value=50000,
        value=0,
        step=500
    )

    loan_amount = st.sidebar.slider(
        "Loan Amount",
        min_value=0,
        max_value=100000,
        value=10000,
        step=1000
    )

    loan_term = st.sidebar.slider(
        "Loan Duration (Months)",
        min_value=0,
        max_value=360,
        value=12,
        step=6
    )

    st.sidebar.markdown("### 🏦 Credit Info")

    credit_history = st.sidebar.selectbox(
        "Credit Score",
        sorted(data["Credit_History"].unique(), reverse=True)
    )

    property_area = st.sidebar.selectbox(
        "Location Type",
        data["Property_Area"].unique()
    )

    # ======================
    # 🔧 IMPORTANT: model safety fix
    # ======================
    # clip extreme values so model doesn't break
    dependents = min(dependents, 3)

    return pd.DataFrame([{
        "Gender": gender,
        "Married": married,
        "Dependents": dependents,
        "Education": education,
        "Self_Employed": self_employed,
        "ApplicantIncome": applicant_income,
        "CoapplicantIncome": coapplicant_income,
        "LoanAmount": loan_amount,
        "Loan_Amount_Term": loan_term,
        "Credit_History": credit_history,
        "Property_Area": property_area
    }])


input_df = user_input()

# =====================
# Main UI Layout
# =====================
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📄 Input Summary")
    st.dataframe(input_df, use_container_width=True)

with col2:
    st.subheader("📊 Prediction Result")

    if st.button("🔮 Predict Loan Status"):

        pred = model.predict(input_df)[0]

        st.markdown("---")

        if pred == "Y":
            st.success("✅ Loan Approved")
        else:
            st.error("❌ Loan Rejected")

        st.info("ℹ️ Prediction is based on trained ML model using historical loan data.")