
def predict_genre(model, vectorizer, description):
    """
    Predict genre of a new movie description
    """
    
    description_tfidf = vectorizer.transform([description])
    prediction = model.predict(description_tfidf)
    
    return prediction[0]