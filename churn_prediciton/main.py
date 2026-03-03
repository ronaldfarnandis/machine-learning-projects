

from train import train_and_evaluate

if __name__ == "__main__":

    dataset_path = r"C:\Users\ronal\OneDrive\Desktop\churn_prediciton\Churn_Modelling.csv"

    train_and_evaluate(
        dataset_path,
        model_type="random_forest"  # logistic, random_forest, gradient_boosting
    )