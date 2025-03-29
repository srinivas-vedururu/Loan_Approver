# Loan Eligibility Prediction Project

## Overview
This project implements a machine learning model to predict loan eligibility based on user-provided financial and personal data. The model uses a dataset (`loan_data.csv`) and employs the XGBoost classifier to determine whether an individual qualifies for a loan. The notebook includes data preprocessing, feature engineering, model training, and a user input system for real-time predictions.

The project is implemented in Python using a Jupyter Notebook and leverages libraries such as `pandas`, `sklearn`, `xgboost`, and `matplotlib` for data manipulation, modeling, and visualization.

## Features
- **Data Preprocessing**: Handles missing data, encodes categorical variables, and scales numerical features.
- **Model Training**: Uses XGBoost to train a classifier on the loan dataset.
- **User Input**: Collects user data interactively with error handling and predicts loan eligibility.
- **Visualization**: Includes plotting functions to evaluate model performance (e.g., confusion matrices).
- **Model Persistence**: Saves the trained model and training data using `joblib`.

## Prerequisites
To run this project, you need the following installed:
- Python 3.6+
- Jupyter Notebook or JupyterLab
- Required Python libraries (install via `pip`):
  ```bash
  pip install pandas sklearn xgboost matplotlib seaborn numpy joblib
  ```

## Dataset
The dataset used is `loan_data.csv`, which contains the following columns:
- `person_age`: Age of the individual
- `person_gender`: Gender (male/female)
- `person_education`: Education level (e.g., High School, Bachelor)
- `person_income`: Annual income
- `person_emp_exp`: Employment experience in years
- `person_home_ownership`: Home ownership status (RENT, OWN, MORTGAGE)
- `loan_amnt`: Loan amount requested
- `loan_intent`: Purpose of the loan (e.g., PERSONAL, EDUCATION)
- `loan_int_rate`: Interest rate of the loan
- `loan_percent_income`: Loan amount as a percentage of income
- `cb_person_cred_hist_length`: Credit history length
- `credit_score`: Credit score of the individual
- `previous_loan_defaults_on_file`: Whether the individual has defaulted on a loan (Yes/No)
- `loan_status`: Target variable (1 = approved, 0 = not approved)

The dataset contains 45,000 entries with no duplicates and no missing values.

## Project Structure
- **Jupyter Notebook**: `loan_approval.ipynb` - Main file containing the code.
- **Dataset**: `loan_data.csv` 
- **Saved Models/Data**: 
  - `model.joblib`: Trained XGBoost model.
  - `x_train.joblib`: Training features.
  - `y_train.joblib`: Training labels.


4. **Run the Notebook**:
   - Execute all cells sequentially.
   - When prompted, provide user inputs (e.g., income, loan amount, credit score) to get a loan eligibility prediction.

## Usage
- **Exploratory Data Analysis (EDA)**: The notebook begins with loading and inspecting the dataset.
- **Preprocessing**: Categorical variables are one-hot encoded, and numerical features are scaled using `MinMaxScaler`.
- **Model Training**: An XGBoost classifier is trained on a subset of features with the highest correlation to `loan_status`.
- **Prediction**: Users input their details (e.g., income, loan amount, credit score), and the model predicts eligibility.
- **Output**: The result is displayed as "Congrats You Are Eligible For Loan" or "Sorry You Are Not Eligible For Loan".

### Example Input
```
Do you have any previous loan defaults on file? (yes/no): No
What is your home ownership type (RENT / OWN / MORTGAGE)? RENT
What is your yearly income? 36991
How much money do you want to borrow? 10000
What is your credit score? 693
```
**Output**:
```
User Input Summary:
{'previous_loan_defaults_on_file': 0, 'person_home_ownership': 'RENT', 'person_income': 36991.0, 'loan_amnt': 10000.0, 'credit_score': 693.0, 'loan_percent_income': 0.27, 'loan_int_rate': 11.66}
Congrats You Are Eligible For Loan
```

## Key Components
- **Feature Selection**: Uses correlation to select the most impactful features (`previous_loan_defaults_on_file`, `loan_percent_income`, etc.).
- **Error Handling**: Ensures valid user inputs (e.g., yes/no, positive numbers).
- **Loan Interest Rate Calculation**: Derived using a custom formula:  
  `loan_int_rate = 18 - (0.015 * credit_score) + 15 * (loan_amnt / person_income)`.

## Future Improvements
- Add cross-validation for better model evaluation.
- Incorporate additional features or datasets for improved accuracy.
- Deploy the model as a web application using Flask or Streamlit.
- Include hyperparameter tuning for the XGBoost model.


## Acknowledgments
- Uses open-source libraries: `pandas`, `sklearn`, `xgboost`, and more.
<h1 align="center"><b>Thank You</b></h1>
