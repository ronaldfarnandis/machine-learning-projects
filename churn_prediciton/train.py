

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from data_loader import load_data
from preprocessing import preprocess_data
from models import get_model


def train_and_evaluate(file_path, model_type="random_forest"):

    print("Loading dataset...")
    df = load_data(file_path)

    print("Preprocessing data...")
    X_train, X_test, y_train, y_test = preprocess_data(df)

    print(f"Training {model_type} model...")
    model = get_model(model_type)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print("\nAccuracy:", accuracy_score(y_test, y_pred))
    print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("\nClassification Report:\n")
    print(classification_report(y_test, y_pred))

    return model