

from train import train_and_evaluate

if __name__ == "__main__":

    train_file = r"C:\Users\ronal\OneDrive\Desktop\fraud_detection\fraudTrain.csv"
    test_file = r"C:\Users\ronal\OneDrive\Desktop\fraud_detection\fraudTest.csv"

    train_and_evaluate(
        train_file,
        test_file,
        model_type="random_forest"
    )