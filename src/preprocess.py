from sklearn.preprocessing import LabelEncoder

def preprocess_data(df):

    encoder = LabelEncoder()

    categorical_cols = df.select_dtypes(
        include='object'
    ).columns

    for col in categorical_cols:
        df[col] = encoder.fit_transform(df[col])

    X = df.drop("Churn", axis=1)
    y = df["Churn"]

    return X , y     