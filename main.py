from src.train import train_model
from src.evaluate import evaluate_model

model, accuracy, features, X_test, y_test = train_model()

print(f"\nModel Accuracy: {accuracy:.2%}")

evaluate_model(
    model,
    X_test,
    y_test
)
