
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

def get_model(model_type="random_forest"):

    if model_type == "logistic":
        return LogisticRegression(max_iter=1000)

    elif model_type == "decision_tree":
        return DecisionTreeClassifier()

    elif model_type == "random_forest":
        return RandomForestClassifier(n_estimators=100)

    else:
        raise ValueError("Choose: logistic, decision_tree, random_forest")