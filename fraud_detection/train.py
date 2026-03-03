

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from data_loader import load_data
from models import get_model


def train_and_evaluate(train_path, test_path, model_type="random_forest"):

    print("Loading training dataset...")
    train_df = load_data(train_path)

    print("Loading test dataset...")
    test_df = load_data(test_path)

    train_df = train_df.select_dtypes(include=["number"])
    test_df = test_df.select_dtypes(include=["number"])

    X_train = train_df.drop("is_fraud", axis=1)
    y_train = train_df["is_fraud"]

    X_test = test_df.drop("is_fraud", axis=1)
    y_test = test_df["is_fraud"]

    print(f"Training {model_type} model...")
    model = get_model(model_type)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print("\nAccuracy:", accuracy_score(y_test, y_pred))
    print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("\nClassification Report:\n")
    print(classification_report(y_test, y_pred))

    return model