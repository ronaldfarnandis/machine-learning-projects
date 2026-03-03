
import pandas as pd

def load_data(file_path):
    """
    Load credit card fraud dataset
    """
    df = pd.read_csv(file_path)
    return df