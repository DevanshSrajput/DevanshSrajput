import streamlit as st
import pandas as pd
import numpy as np
import joblib

# --- Page Configuration ---
st.set_page_config(
    page_title="Loan Eligibility Predictor",
    page_icon="🏦",
    layout="centered",
    initial_sidebar_state="auto"
)

# --- Custom CSS for Advanced Styling ---
st.markdown("""
    <style>
        /* Keyframes for fade-in animation */
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(-20px); }
            to { opacity: 1; transform: translateY(0); }
        }

        /* Main app background with gradient */
        .stApp {
            background: linear-gradient(to right top, #d4e4f7, #f5f8fc); /* <-- RE-INTRODUCED VISIBLE GRADIENT */
            font-family: 'Helvetica Neue', sans-serif;
        }
        
        /* Main title */
        h1 {
            color: #1e3a8a; /* Deep blue */
            text-align: center;
            font-weight: 700;
        }
        .subtitle {
            text-align: center;
            font-size: 1.15em;
            color: #4b5563; /* Gray */
            margin-bottom: 25px;
        }

        /* Form container with enhanced shadow and animation */
        [data-testid="stForm"] {
            background: #009698;
            padding: 25px 30px;
            border-radius: 18px;
            box-shadow: 0 12px 28px rgba(0,0,0,0.08);
            border: 1px solid #e6e6e6;
            animation: fadeIn 0.5s ease-out;
            color: #333 !important; /* General text fix */
        }

        /* --- NEW FIXES START --- */
        /* Force widget LABELS (like "Gender", "Applicant Income") to be dark */
        [data-testid="stForm"] div[data-testid="stWidgetLabel"] {
            color: #333 !important;
            opacity: 1 !important; /* Ensure it's not faded */
        }
        
        /* Force RADIO BUTTON options (like "Yes", "No") to be dark */
        [data-testid="stForm"] .stRadio label {
            color: #333 !important;
        }
        /* --- NEW FIXES END --- */


        /* Tab styles */
        .stTabs [data-baseweb="tab-list"] {
            justify-content: center;
        }
        .stTabs [data-baseweb="tab"] {
            padding: 12px 18px;
            font-size: 1.05em;
            font-weight: 600;
            color: #4b5563;
        }
        .stTabs [data-baseweb="tab--selected"] {
            color: #0068c9;
            border-bottom-color: #0068c9;
        }

        /* Ensure tab content also has dark text */
        .stTabs [data-baseweb="tab-panel"] {
             color: #333 !important;
        }

        /* Button style with gradient */
        .stButton>button {
            color: #ffffff !important;
            background-image: linear-gradient(to right, #0068c9 0%, #007bff 51%, #0068c9 100%);
            background-size: 200% auto;
            border: none;
            border-radius: 10px;
            padding: 14px 30px;
            font-size: 17px;
            font-weight: bold;
            width: 100%;
            transition: 0.5s;
            box-shadow: 0 4px 15px rgba(0, 104, 201, 0.2);
        }
        .stButton>button:hover {
            background-position: right center; /* change the direction of the change here */
            color: #ffffff !important;
            text-decoration: none;
            box-shadow: 0 6px 20px rgba(0, 104, 201, 0.3);
            transform: translateY(-2px);
        }

        /* Custom Result Boxes */
        .result-box {
            display: flex;
            align-items: center;
            padding: 20px;
            border-radius: 12px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.05);
            margin-top: 25px;
            color: #000000 !important;
        }
        .result-box.success {
            background: linear-gradient(to right, #e6f7ec, #d1f0db);
            border: 1px solid #5cb85c;
        }
        .result-box.error {
            background: linear-gradient(to right, #fdecea, #f9d5d2);
            border: 1px solid #d9534f;
        }
        .result-icon {
            font-size: 2.5em;
            margin-right: 20px;
        }
        .result-text strong {
            font-size: 1.3em;
            display: block;
        }
        .result-metric {
            margin-left: auto;
            text-align: center;
        }
        .result-metric span {
            font-size: 2.2em !important;
            font-weight: 700 !important;
        }
        .result-metric div {
            font-weight: 600;
            color: #4b5563;
        }
        .success .result-metric span { color: #2e7d32; }
        .error .result-metric span { color: #d32f2f; }
    </style>
""", unsafe_allow_html=True)

# --- Load Model and Columns ---
try:
    model = joblib.load('loan_predictor_model.joblib')
    model_columns = joblib.load('model_columns.joblib')
except FileNotFoundError:
    st.error("Model files not found. Please run `train_model.py` first to generate the model.")
    st.stop()
except Exception as e:
    st.error(f"An error occurred while loading model files: {e}")
    st.stop()

# --- Main Application ---
st.markdown("<h1 style='text-align: center; color: #1e3a8a;'>🏦 Loan Eligibility Predictor</h1>", unsafe_allow_html=True)
# This line was indented, causing the error. It is now fixed.
st.markdown("<p class='subtitle'>Fill in the details to get an instant eligibility check</p>", unsafe_allow_html=True)

# --- Prediction Form with Tabs ---
with st.form("loan_form"):
    
    tab1, tab2, tab3 = st.tabs([
        "👤 Personal Information", 
        "💼 Financial Information", 
        "🏦 Loan Details"
    ])

    with tab1:
        st.write("") # Add a little space
        col1, col2 = st.columns(2)
        with col1:
            gender = st.selectbox("Gender", ("Male", "Female"))
            dependents = st.selectbox("Number of Dependents", ("0", "1", "2", "3+"))
            self_employed = st.selectbox("Self Employed", ("No", "Yes"))
        with col2:
            married = st.selectbox("Married", ("No", "Yes"))
            education = st.selectbox("Education", ("Graduate", "Not Graduate"))
            property_area = st.selectbox("Property Area", ("Urban", "Semiurban", "Rural"))

    with tab2:
        st.write("")
        col3, col4 = st.columns(2)
        with col3:
            applicant_income = st.number_input("Applicant Income ($)", min_value=0, step=100,
                                               help="Enter the applicant's monthly income.")
        with col4:
            coapplicant_income = st.number_input("Co-applicant Income ($)", min_value=0, step=100,
                                                 help="Enter '0' if there is no co-applicant.")

    with tab3:
        st.write("")
        col5, col6 = st.columns(2)
        with col5:
            loan_amount = st.number_input("Loan Amount (in thousands $)", min_value=0, step=1,
                                          help="Enter the total loan amount requested (e.g., 150 for $150,000).")
            loan_amount_term = st.number_input("Loan Amount Term (in months)", min_value=0, step=12,
                                               help="Enter the loan repayment term in months (e.g., 360 for 30 years).")
        with col6:
            st.write("") # Align with other inputs
            st.write("")
            credit_history = st.radio("Credit History Available?", ("Yes", "No"),
                                      help="Does the applicant have a credit history?")

    # --- Submit Button ---
    st.write("") # Add some space
    submitted = st.form_submit_button("Check Eligibility")

# --- Prediction Logic ---
if submitted:
    # --- Data Preprocessing on Input ---
    credit_history_float = 1.0 if credit_history == "Yes" else 0.0
    input_data = {
        'Gender': [gender], 'Married': [married], 'Dependents': [dependents],
        'Education': [education], 'Self_Employed': [self_employed],
        'ApplicantIncome': [applicant_income], 'CoapplicantIncome': [coapplicant_income],
        'LoanAmount': [loan_amount], 'Loan_Amount_Term': [loan_amount_term],
        'Credit_History': [credit_history_float], 'Property_Area': [property_area]
    }
    
    query_df = pd.DataFrame(input_data)
    query_encoded = pd.get_dummies(query_df, drop_first=True)
    query_aligned = query_encoded.reindex(columns=model_columns, fill_value=0)
    
    try:
        query_aligned = query_aligned.astype(float)
    except Exception as e:
        st.error(f"Error in data conversion: {e}")
        st.stop()

    # --- Make Prediction ---
    try:
        prediction = model.predict(query_aligned)
        probability = model.predict_proba(query_aligned)

        st.subheader("Prediction Result")
        
        # --- Display Custom Result Box ---
        if prediction[0] == 1:
            confidence = probability[0][1] * 100
            st.markdown(f"""
                <div class="result-box success">
                    <div class="result-icon">✅</div>
                    <div class="result-text">
                        <strong>Status: Loan Approved</strong>
                        Based on the provided details, the application is likely to be approved.
                    </div>
                    <div class="result-metric">
                        <div>Confidence</div>
                        <span>{confidence:.2f}%</span>
                    </div>
                </div>
            """, unsafe_allow_html=True)
        
        else:
            confidence = probability[0][0] * 100
            st.markdown(f"""
                <div class="result-box error">
                    <div class="result-icon">❌</div>
                    <div class="result-text">
                        <strong>Status: Loan Rejected</strong>
                        Based on the provided details, the application is likely to be rejected.
                    </div>
                    <div class="result-metric">
                        <div>Confidence</div>
                        <span>{confidence:.2f}%</span>
                    </div>
                </div>
            """, unsafe_allow_html=True)

    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
