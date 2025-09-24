import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load the pre-trained model and the column list
try:
    model = joblib.load('loan_predictor_model.joblib')
    model_columns = joblib.load('model_columns.joblib')
except FileNotFoundError:
    st.error("Model files not found. Please run `train_model.py` first to generate the model.")
    st.stop()

def main():
    st.set_page_config(page_title="Loan Eligibility Predictor", page_icon="💰", layout="wide")
    
    # Custom CSS for a better look
    st.markdown("""
        <style>
            .reportview-container {
                background: #f0f2f6;
            }
            .sidebar .sidebar-content {
                background: #f0f2f6;
            }
            .stButton>button {
                color: white;
                background-color: #4CAF50;
                border-radius:10px;
                border: none;
                padding: 10px 24px;
                text-align: center;
                text-decoration: none;
                display: inline-block;
                font-size: 16px;
                margin: 4px 2px;
                cursor: pointer;
            }
            .st-success {
                color: #2e7d32;
            }
            .st-error {
                color: #d32f2f;
            }
        </style>
    """, unsafe_allow_html=True)

    st.title("💰 Loan Eligibility Predictor")
    st.markdown("Enter the applicant's details to predict whether a loan will be approved or not.")

    # Create input fields for user data
    with st.form("loan_prediction_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            gender = st.selectbox("Gender", ["Male", "Female"])
            married = st.selectbox("Married", ["Yes", "No"])
            dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])
            education = st.selectbox("Education", ["Graduate", "Not Graduate"])
            
        with col2:
            self_employed = st.selectbox("Self Employed", ["Yes", "No"])
            applicant_income = st.number_input("Applicant Income", min_value=0, value=5000)
            coapplicant_income = st.number_input("Coapplicant Income", min_value=0, value=1500)
            loan_amount = st.number_input("Loan Amount (in thousands)", min_value=0, value=150)

        with col3:
            loan_amount_term = st.number_input("Loan Amount Term (in months)", min_value=12, value=360, step=12)
            credit_history = st.selectbox("Credit History", [1.0, 0.0], format_func=lambda x: 'Yes' if x == 1.0 else 'No')
            property_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])

        submit_button = st.form_submit_button(label="Predict Eligibility")

    if submit_button:
        # Create a dictionary from the user inputs
        input_data = {
            'Gender': [gender],
            'Married': [married],
            'Dependents': [dependents],
            'Education': [education],
            'Self_Employed': [self_employed],
            'ApplicantIncome': [applicant_income],
            'CoapplicantIncome': [coapplicant_income],
            'LoanAmount': [loan_amount],
            'Loan_Amount_Term': [loan_amount_term],
            'Credit_History': [credit_history],
            'Property_Area': [property_area]
        }
        
        # Create a pandas DataFrame from the input data
        query_df = pd.DataFrame(input_data)
        
        # One-hot encode the categorical variables
        query_encoded = pd.get_dummies(query_df, drop_first=True)
        
        # Align the columns of the query DataFrame with the training columns
        query_aligned = query_encoded.reindex(columns=model_columns, fill_value=0)

        # Make prediction
        prediction = model.predict(query_aligned)
        probability = model.predict_proba(query_aligned)

        st.subheader("Prediction Result:")
        if prediction[0] == 1:
            st.success(f"Congratulations! The loan is likely to be **Approved**.")
            st.info(f"Confidence Score: **{probability[0][1]*100:.2f}%**")
        else:
            st.error(f"Unfortunately, the loan is likely to be **Rejected**.")
            st.info(f"Confidence Score: **{probability[0][0]*100:.2f}%**")

if __name__ == '__main__':
    main()
