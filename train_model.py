import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

def train():
    # Load the training data
    train_df = pd.read_csv('train_u6lujuX_CVtuZ9i.csv')

    # --- Data Cleaning and Preprocessing ---

    # Drop the Loan_ID column as it is not relevant for prediction
    train_df = train_df.drop('Loan_ID', axis=1)

    # Impute missing values in numerical columns with the mean
    for col in ['LoanAmount', 'Loan_Amount_Term', 'Credit_History']:
        train_df[col].fillna(train_df[col].mean(), inplace=True)

    # Impute missing values in categorical columns with the mode
    for col in ['Gender', 'Married', 'Dependents', 'Self_Employed']:
        train_df[col].fillna(train_df[col].mode()[0], inplace=True)

    # --- Feature Engineering ---

    # Convert categorical features into dummy variables (one-hot encoding)
    train_df = pd.get_dummies(train_df, drop_first=True)

    # --- Model Training ---

    # Separate features (X) and target (y)
    X = train_df.drop('Loan_Status_Y', axis=1)
    y = train_df['Loan_Status_Y']

    # Split the data for validation (optional but good practice)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize and train the Logistic Regression model
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)

    # --- Evaluation ---
    y_pred = model.predict(X_val)
    print(f"Validation Accuracy: {accuracy_score(y_val, y_pred):.4f}")

    # --- Save the Model and Columns ---
    
    # Save the trained model to a file
    joblib.dump(model, 'loan_predictor_model.joblib')
    print("Model saved as loan_predictor_model.joblib")

    # Save the list of columns used for training
    model_columns = list(X.columns)
    joblib.dump(model_columns, 'model_columns.joblib')
    print("Model columns saved as model_columns.joblib")


if __name__ == '__main__':
    train()
