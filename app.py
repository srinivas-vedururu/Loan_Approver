import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from xgboost import XGBClassifier
import joblib  

st.title("Loan Eligibility Prediction")
df = pd.read_csv("loan_data.csv")

# Input fields
previous_loan_defaults = st.radio("Do you have any previous loan defaults on file?", ["Yes", "No"])
person_home_ownership = st.selectbox("What is your home ownership type?", ["RENT", "OWN", "MORTGAGE"])
person_income = st.number_input("What is your yearly income?", min_value=0.0, format="%.2f")
loan_amnt = st.number_input("How much money do you want to borrow?", min_value=0.0, format="%.2f")
credit_score = st.number_input("What is your credit score?", min_value=0, max_value=850)

# Convert inputs
previous_loan_defaults = 1 if previous_loan_defaults == "Yes" else 0

if person_income > 0:
    loan_percent_income = round((loan_amnt / person_income), 2)
    loan_int_rate = round(18 - (0.015 * credit_score) + 15 * (loan_amnt / person_income), 2)
else:
    loan_percent_income = 0  # Default value to avoid division by zero
    loan_int_rate = round(18 - (0.015 * credit_score), 2)  # Only consider credit score if income is 0

# Create input dictionary
single_input = {
    'previous_loan_defaults_on_file': previous_loan_defaults,
    'person_home_ownership': person_home_ownership,
    'person_income': person_income,
    'loan_amnt': loan_amnt,
    'credit_score': credit_score,
    'loan_percent_income': loan_percent_income,
    'loan_int_rate': loan_int_rate
}

# Convert to DataFrame
single_input_df = pd.DataFrame([single_input])
single_input_num_cols = ['previous_loan_defaults_on_file', 'person_income', 'loan_amnt', 'credit_score', 'loan_percent_income', 'loan_int_rate']
single_input_cat_col = 'person_home_ownership'

# One-Hot Encoding
encoder = OneHotEncoder(sparse_output=False,handle_unknown="ignore")
encoder.fit(pd.DataFrame(single_input_df["person_home_ownership"]))
encoded_cols = list(encoder.get_feature_names_out())
single_input_df[encoded_cols] = encoder.transform(pd.DataFrame(single_input_df["person_home_ownership"]))

# Scaling
scaler = MinMaxScaler()
scaler.fit(df[single_input_num_cols].replace({'Yes': 1, 'No': 0}))  
single_input_df[single_input_num_cols] = scaler.transform(single_input_df[single_input_num_cols])
single_input_df = single_input_df[single_input_num_cols+encoded_cols]

# Load model
X_train = joblib.load("x_train.joblib")
Y_train = joblib.load("y_train.joblib")
model = XGBClassifier(n_jobs=-1,random_state=7)
model.fit(X_train[single_input_num_cols+encoded_cols],Y_train)
predction = model.predict(single_input_df)

if st.button("Check Eligibility"):
    prediction = model.predict(single_input_df)
    if prediction == 0:
        st.error("Sorry, You Are Not Eligible for a Loan.")
    else:
        st.success("Congrats! You Are Eligible for a Loan.")
