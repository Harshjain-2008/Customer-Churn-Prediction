import pandas as pd 

def clean_data(df):

    # remove duplicates 
    df = df.drop_duplicates()

    # Remove customer Id 
    if "customerID" in df.columns:
        df.drop("customerID", axis=1,inplace=True)

    # Handel Charges 
    df["TotalCharges"] = df["TotalCharges"].replace(" ", pd.NA)

    df["TotalCharges"]  = pd.to_numeric(df["TotalCharges"], errors="coerce")

    df["TotalCharges"].fillna(df["TotalCharges"].median(), inplace=True)

    return df

    
