import joblib 

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

from src.clean_data import clean_data
from src.preprocess import preprocess_data
from src.data_loader import load_data

def train_model():

    df = load_data("data/raw/customer_churn.csv")

    df = clean_data(df)

    X, y = preprocess_data(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )

    model.fit(X_train,y_train)

    pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, pred)

    joblib.dump(model, "models/churn_model.pkl")

    return model, accuracy,X.columns,X_test,y_test