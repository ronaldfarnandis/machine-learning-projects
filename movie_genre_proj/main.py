

from train import train_and_evaluate
from predict import predict_genre


if __name__ == "__main__":
    
    train_file = "train_data.txt"
    test_file = "test_data_solution.txt"
    
    model, vectorizer = train_and_evaluate(
        train_file,
        test_file,
        model_type="svm"  
    )
    
    # Example prediction
    sample_plot = "A brave warrior fights dragons to save his kingdom."
    
    predicted = predict_genre(model, vectorizer, sample_plot)
    
    print("\nPredicted Genre:", predicted)