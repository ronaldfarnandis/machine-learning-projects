
from sklearn.feature_extraction.text import TfidfVectorizer

def create_vectorizer():
    """
    Create and return TF-IDF vectorizer
    """
    return TfidfVectorizer(
        stop_words="english",
        max_df=0.8
    )

def transform_data(vectorizer, X_train, X_test):
    """
    Fit on training data and transform both train and test
    """
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    
    return X_train_tfidf, X_test_tfidf