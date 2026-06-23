import joblib
import pandas as pd 
import matplotlib.pyplot as plt
import streamlit as st

model = joblib.load("models/churn_model.pkl")

st.title("CUSTOMER CHURN PREDICTION")

st.sidebar.title("Model Information")
st.sidebar.metric("Accuracy", "79.3")

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
    by=importance_df,
    ascending=False
)

fig, ax = plt.subplot(figsize=(8,5))

ax.brah(
    importance_df["Features"][:10],
    importance_df["Importance"][:10]
)

ax.invert_yaxis()

st.pyplot(fig)


