import pandas as pd
import numpy as np
import re
import joblib
import pickle
from sklearn.preprocessing import StandardScaler, RobustScaler, OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer

# Median & Mode for imputation
GLOBAL_STATS = {
    'Age': 33.00,
    'Annual_Income': 37182.62,
    'Num_Bank_Accounts': 5.00,
    'Num_Credit_Card': 5.00,
    'Interest_Rate': 13.00,
    'Num_of_Loan': 3.00,
    'Delay_from_due_date': 18.00,
    'Num_of_Delayed_Payment': 14.00,
    'Changed_Credit_Limit': 9.36,
    'Num_Credit_Inquiries': 5.00,
    'Outstanding_Debt': 1161.10,
    'Credit_Utilization_Ratio': 32.30,
    'Total_EMI_per_month': 66.49,
    'Amount_invested_monthly': 127.76,
    'Monthly_Balance': 344.82,
    'Credit_History_Age_in_Months': 218.00,
    'Occupation': 'Lawyer',
    'Credit_Mix': 'Standard',
    'Payment_of_Min_Amount': 'Yes',
    'Payment_Behaviour': 'Low_spent_Small_value_payments'
}

def numerical_cleaning(data):
    if pd.isna(data) or type(data) is not str:
        return data
    
    cleaned_data = re.sub(r'[^0-9.-]', '', data)
    
    if cleaned_data == '':
        return np.nan
    else:
        return cleaned_data

def convert_age_to_months(age_str):
    if pd.isna(age_str):
        return np.nan
    
    parts = age_str.split()
    years = int(parts[0])
    months = int(parts[3])
    return (years * 12) + months

def clean_categorical_values(df):
    df = df.copy()
    
    # Clean categorical columns
    df["Payment_Behaviour"] = df["Payment_Behaviour"].replace("!@9#%8", np.nan)
    df["Occupation"] = df["Occupation"].replace("_______", np.nan)
    df["Credit_Mix"] = df["Credit_Mix"].replace("_", np.nan)
    df["Payment_of_Min_Amount"] = df["Payment_of_Min_Amount"].replace("NM", np.nan)
    
    return df

def preprocess_single_input(input_dict, transformer_path="preprocessing/transformer_data.pkl"):
    df = pd.DataFrame([input_dict])
    
    # Dummy Customer_ID
    df['Customer_ID'] = 'NEW_USER_001'
    
    df = preprocess_data(df, is_training=False)
    transformer = joblib.load(transformer_path)
    
    # Drop Customer_ID and other columns not used in training
    feature_columns = df.columns.drop(['Customer_ID'], errors='ignore')
    X = df[feature_columns]
    
    X_transformed = transformer.transform(X)
    
    return X_transformed

def preprocess_data(df, is_training=True, transformer_path=None):
    df = df.copy()
    
    # Remove unnecessary columns
    columns_to_drop = ['ID', 'Month', 'Name', 'SSN', 'Monthly_Inhand_Salary', 'Type_of_Loan']
    df = df.drop(columns=[col for col in columns_to_drop if col in df.columns])
    
    int_cols = ['Age', 'Num_of_Loan', 'Num_of_Delayed_Payment']
    float_cols = ['Annual_Income', 'Changed_Credit_Limit', 'Outstanding_Debt', 
                  'Amount_invested_monthly', 'Monthly_Balance']
    
    # Clean numerical values
    cols_to_clean = int_cols + float_cols
    for col_name in cols_to_clean:
        if col_name in df.columns:
            df[col_name] = df[col_name].apply(numerical_cleaning)
    
    for col in cols_to_clean:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    if 'Credit_History_Age' in df.columns:
        df['Credit_History_Age_in_Months'] = df['Credit_History_Age'].apply(convert_age_to_months)
        df['Credit_History_Age_in_Months'] = df['Credit_History_Age_in_Months'].fillna(0)
        df = df.drop(columns=['Credit_History_Age'])
    
    # Clean categorical values
    df = clean_categorical_values(df)
    
    # Handle missing values
    df = handle_missing_values_inference(df)
    
    return df

def handle_missing_values_inference(df):
    df = df.copy()
    
    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
    for col in numerical_cols:
        if col in GLOBAL_STATS:
            df[col] = df[col].fillna(GLOBAL_STATS[col])
    
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        if col in GLOBAL_STATS and col != 'Customer_ID':
            df[col] = df[col].fillna(GLOBAL_STATS[col])
    
    return df

def validate_input_columns(df, required_columns=None):
    if required_columns is None:
        required_columns = [
            'Age', 'Annual_Income', 'Num_Bank_Accounts', 'Num_Credit_Card', 
            'Interest_Rate', 'Num_of_Loan', 'Delay_from_due_date', 
            'Num_of_Delayed_Payment', 'Changed_Credit_Limit', 'Num_Credit_Inquiries',
            'Outstanding_Debt', 'Credit_Utilization_Ratio', 'Total_EMI_per_month',
            'Amount_invested_monthly', 'Monthly_Balance', 'Occupation',
            'Credit_Mix', 'Payment_of_Min_Amount', 'Payment_Behaviour'
        ]
    
    missing_columns = [col for col in required_columns if col not in df.columns]
    is_valid = len(missing_columns) == 0
    
    return is_valid, missing_columns

def preprocess_batch_data(df, transformer_path="preprocessing/transformer_data.pkl"):
    is_valid, missing_cols = validate_input_columns(df)
    if not is_valid:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Add dummy Customer_ID if not present
    if 'Customer_ID' not in df.columns:
        df = df.copy()
        df['Customer_ID'] = [f'BATCH_USER_{i:03d}' for i in range(len(df))]
    
    df_processed = preprocess_data(df, is_training=False)
    transformer = joblib.load(transformer_path)
    
    # Drop Customer_ID and other columns not used in training
    feature_columns = df_processed.columns.drop(['Customer_ID'], errors='ignore')
    X = df_processed[feature_columns]
    
    X_transformed = transformer.transform(X)
    
    return X_transformed, df_processed

def get_feature_names():
    return [
        'Age', 'Annual_Income', 'Num_Bank_Accounts', 'Num_Credit_Card', 
        'Interest_Rate', 'Num_of_Loan', 'Delay_from_due_date', 
        'Num_of_Delayed_Payment', 'Changed_Credit_Limit', 'Num_Credit_Inquiries',
        'Outstanding_Debt', 'Credit_Utilization_Ratio', 'Total_EMI_per_month',
        'Amount_invested_monthly', 'Monthly_Balance', 'Credit_History_Age',
        'Occupation', 'Credit_Mix', 'Payment_of_Min_Amount', 'Payment_Behaviour'
    ]

def create_sample_input():
    return {
        'Age': 33,
        'Annual_Income': 37182.62,
        'Num_Bank_Accounts': 5,
        'Num_Credit_Card': 5,
        'Interest_Rate': 13,
        'Num_of_Loan': 3,
        'Delay_from_due_date': 18,
        'Num_of_Delayed_Payment': 14,
        'Changed_Credit_Limit': 9.36,
        'Num_Credit_Inquiries': 5,
        'Outstanding_Debt': 1161.10,
        'Credit_Utilization_Ratio': 32.30,
        'Credit_History_Age': '18 Years and 2 Months',
        'Total_EMI_per_month': 66.49,
        'Amount_invested_monthly': 127.76,
        'Monthly_Balance': 344.82,
        'Occupation': 'Lawyer',
        'Credit_Mix': 'Standard',
        'Payment_of_Min_Amount': 'Yes',
        'Payment_Behaviour': 'Low_spent_Small_value_payments'
    }