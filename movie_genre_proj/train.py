
from sklearn.metrics import accuracy_score, classification_report
from data_loader import load_data
from vectorizer import create_vectorizer, transform_data
from models import get_model


def train_and_evaluate(train_path, test_path, model_type="svm"):

    print("Loading training data...")
    train_df = load_data(train_path)

    print("Loading test data...")
    test_df = load_data(test_path)

    X_train = train_df["description"]
    y_train = train_df["genre"]

    X_test = test_df["description"]
    y_test = test_df["genre"]

    # Vectorization
    vectorizer = create_vectorizer()
    X_train_tfidf, X_test_tfidf = transform_data(
        vectorizer, X_train, X_test
    )

    # Model
    model = get_model(model_type)
    print(f"Training {model_type} model...")
    model.fit(X_train_tfidf, y_train)

    # Prediction
    y_pred = model.predict(X_test_tfidf)

    # Evaluation
    print("\nAccuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:\n")
    print(classification_report(y_test, y_pred))

    return model, vectorizer