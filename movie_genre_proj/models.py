
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB


def get_model(model_type="svm"):
    """
    Returns a machine learning model based on model_type.

    Available options:
    - "logistic"
    - "svm"
    - "naive_bayes"
    """

    if model_type == "logistic":
        return LogisticRegression(max_iter=1000)

    elif model_type == "svm":
        return LinearSVC()

    elif model_type == "naive_bayes":
        return MultinomialNB()

    else:
        raise ValueError(
            "Invalid model_type. Choose from: 'logistic', 'svm', 'naive_bayes'"
        )