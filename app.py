import joblib
import pandas as pd 
import matplotlib.pyplot as plt
import streamlit as st

model = joblib.load("models/churn_model.pkl")

st.title("CUSTOMER CHURN PREDICTION")


# Sidebar 
st.sidebar.title("📌 Model Information")

st.sidebar.metric(
    "Model Accuracy",
    "81.5%"
)

st.sidebar.markdown("---")

st.sidebar.subheader("🤖 About Model")

st.sidebar.info(
    """
    Model Used: Random Forest Classifier

    Purpose:
    Predict whether a customer is likely to churn (leave the service) or stay.

    Target Variable:
    Churn (Yes/No)

    Dataset:
    IBM Telco Customer Churn Dataset

    Algorithm Strengths:
    - Handles categorical and numerical data well
    - Reduces overfitting
    - Provides feature importance scores
    - High prediction accuracy
    """
)

st.sidebar.markdown("---")

st.sidebar.subheader("📊 Project Workflow")

st.sidebar.markdown("""
1. Data Collection
2. Data Cleaning
3. Exploratory Data Analysis
4. Data Preprocessing
5. Model Training
6. Model Evaluation
7. Prediction
""")

# User inputs 

tenure = st.slider(
    "Tenure (Months)",
    1,
    72,
    12
)

monthly = st.number_input(
    "Monthly Charges",
    0.0,
    200.0,
    50.0
)

total_charges = st.number_input(
    "Total Charges",
    0.0,
    10000.0,
    500.0
)

senior = st.selectbox(
    "Senior Citizen ",
    [0,1]
)

partner = st.selectbox(
    "Partner",
    ["Yes", "No"]
)

dependents = st.selectbox(
    "Dependents",
    ["Yes","NO"]
)

# Prediction 

if st.button("Predict"):

    sample = pd.DataFrame([{
        'gender': 1,
        'SeniorCitizen': senior,
        'Partner': 1 if partner == "Yes" else 0,
        'Dependents': 1 if dependents == "Yes" else 0,
        'tenure': tenure,
        'PhoneService': 1,
        'MultipleLines': 0,
        'InternetService': 1,
        'OnlineSecurity': 0,
        'OnlineBackup': 0,
        'DeviceProtection': 0,
        'TechSupport': 0,
        'StreamingTV': 0,
        'StreamingMovies': 0,
        'Contract': 0,
        'PaperlessBilling': 1,
        'PaymentMethod': 0,
        'MonthlyCharges': monthly,
        'TotalCharges': total_charges
    }])
     
    prediction = model.predict(sample)[0]
    probability = model.predict_proba(sample)

    if prediction==1:
        st.error(
            f"Customer Likely to Churn\n\n"
            f"Probability: {probability[0][1]*100:.2f}%"
        ) 
    else:
        st.success(
            f"Customer Likely to Stay\n\n "
            f"Probability: {probability[0][1]*100:.2f}%"
        )    


# Feature Importance 
st.subheader("Feature importance")

importance = model.feature_importances_
features = sample.columns

importance_df = pd.DataFrame({
    "Feature": features,
    "Importance": importance
})

importance_df = importance_df.sort_values(
    by="Importance",
    ascending=False
)

fig, ax = plt.subplots(figsize=(8,5))

ax.barh(
    importance_df["Feature"][:10],
    importance_df["Importance"][:10]
)

ax.invert_yaxis()

st.pyplot(fig)


