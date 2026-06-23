import joblib 
import pandas as pd 

model = joblib.load("models/churn_model.pkl")

def predict_customer(data):

    sample = pd.DataFrame([data])

    predictions = model.predict(sample)

    probability = model.predict_proba(sample)

    return predictions[0], probability

 