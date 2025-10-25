import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, roc_auc_score

# --- Page Configuration ---
st.set_page_config(page_title="Analysis Dashboard", page_icon="📊", layout="wide")

# --- Caching Functions for Performance ---
@st.cache_data
def load_data():
    df = pd.read_csv('train_u6lujuX_CVtuZ9i.csv')
    df = df.drop('Loan_ID', axis=1)
    return df

@st.cache_data
def preprocess_data(df):
    df_processed = df.copy()
    for col in ['LoanAmount', 'Loan_Amount_Term']:
        df_processed[col].fillna(df_processed[col].mean(), inplace=True)
    for col in ['Gender', 'Married', 'Dependents', 'Self_Employed', 'Credit_History']:
        df_processed[col].fillna(df_processed[col].mode()[0], inplace=True)
    df_processed = pd.get_dummies(df_processed, drop_first=True)
    X = df_processed.drop('Loan_Status_Y', axis=1)
    y = df_processed['Loan_Status_Y']
    scaler = StandardScaler()
    numerical_features = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term']
    X[numerical_features] = scaler.fit_transform(X[numerical_features])
    return X, y

# --- Main Application ---
st.title("📊 Model Training & Analysis Dashboard")
st.markdown("This page details the project's analysis, from data exploration to model performance evaluation.")

# --- 1. Exploratory Data Analysis (EDA) ---
st.header("1. Exploratory Data Analysis (EDA)")
train_df = load_data()

st.subheader("Loan Approval Distribution")
st.write(train_df['Loan_Status'].value_counts(normalize=True))

st.subheader("Visualizing Categorical Features vs. Loan Status")
categorical_cols = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Property_Area', 'Credit_History']
fig_eda, axes = plt.subplots(3, 3, figsize=(15, 10))
axes = axes.flatten()
for i, col in enumerate(categorical_cols):
    sns.countplot(x=col, hue='Loan_Status', data=train_df, ax=axes[i])
    axes[i].set_title(f'Loan Status by {col}')
plt.tight_layout()
st.pyplot(fig_eda)

# --- 2. Model Training and Evaluation ---
st.header("2. Model Training & Evaluation")
st.info("The models are trained on an 80/20 split of the dataset. Below are the performance results on the 20% test set.")

X, y = preprocess_data(train_df.copy())
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# (a) Logistic Regression
lr_model = LogisticRegression(max_iter=1000, random_state=42)
lr_model.fit(X_train, y_train)
lr_pred = lr_model.predict(X_test)
lr_proba = lr_model.predict_proba(X_test)[:, 1]

# (b) Random Forest
rf_model = RandomForestClassifier(n_estimators=150, random_state=42)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)
rf_proba = rf_model.predict_proba(X_test)[:, 1]

# --- Performance Metrics ---
st.subheader("Performance Metrics")
lr_accuracy = accuracy_score(y_test, lr_pred)
rf_accuracy = accuracy_score(y_test, rf_pred)
lr_auc = roc_auc_score(y_test, lr_proba)
rf_auc = roc_auc_score(y_test, rf_proba)

col1, col2 = st.columns(2)
with col1:
    st.metric("Logistic Regression Accuracy", f"{lr_accuracy:.4f}")
    st.metric("Logistic Regression AUC", f"{lr_auc:.4f}")
with col2:
    st.metric("Random Forest Accuracy", f"{rf_accuracy:.4f}")
    st.metric("Random Forest AUC", f"{rf_auc:.4f}")

st.subheader("Confusion Matrices")
fig_cm, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
sns.heatmap(confusion_matrix(y_test, lr_pred), annot=True, fmt='d', cmap='Blues', ax=ax1)
ax1.set_title('Logistic Regression')
sns.heatmap(confusion_matrix(y_test, rf_pred), annot=True, fmt='d', cmap='Greens', ax=ax2)
ax2.set_title('Random Forest')
st.pyplot(fig_cm)

st.subheader("ROC Curve Comparison")
lr_fpr, lr_tpr, _ = roc_curve(y_test, lr_proba)
rf_fpr, rf_tpr, _ = roc_curve(y_test, rf_proba)
fig_roc, ax_roc = plt.subplots(figsize=(10, 7))
ax_roc.plot(lr_fpr, lr_tpr, linestyle='--', label=f'Logistic Regression (AUC = {lr_auc:.2f})')
ax_roc.plot(rf_fpr, rf_tpr, marker='.', label=f'Random Forest (AUC = {rf_auc:.2f})')
ax_roc.plot([0, 1], [0, 1], 'k--')
ax_roc.set_title('ROC Curve Comparison')
ax_roc.set_xlabel('False Positive Rate')
ax_roc.set_ylabel('True Positive Rate')
ax_roc.legend()
st.pyplot(fig_roc)

# --- 3. Model Recommendation ---
st.header("3. Model Recommendation")
if lr_accuracy > rf_accuracy:
    best_model_name = "Logistic Regression"
else:
    best_model_name = "Random Forest"

st.success(f"**Recommendation:** The **{best_model_name}** model is recommended and used for the live predictions in the main app.")
