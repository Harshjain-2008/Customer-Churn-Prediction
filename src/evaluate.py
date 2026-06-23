from sklearn.metrics import accuracy_score,classification_report,confusion_matrix

def evaluate_model(model, X_test, y_test):

    pred = model.predict(X_test)

    print("\nACCURACY: ")
    print(accuracy_score(
        y_test,
        pred
    ))

    print("\nClassification Report: ")
    print(classification_report(
        y_test,
        pred
    ))

    print("\nConfusion Matrix: ")
    print(confusion_matrix(
        y_test,
        pred
    ))

    