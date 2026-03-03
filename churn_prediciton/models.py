

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

def get_model(model_type="random_forest"):

    if model_type == "logistic":
        return LogisticRegression(max_iter=1000)

    elif model_type == "random_forest":
        return RandomForestClassifier(n_estimators=100)

    elif model_type == "gradient_boosting":
        return GradientBoostingClassifier()

    else:
        raise ValueError("Choose: logistic, random_forest, gradient_boosting")
    return RandomForestClassifier(
    n_estimators=200,
    class_weight="balanced",
    max_depth=10
)