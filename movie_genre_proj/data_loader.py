

import pandas as pd

def load_data(file_path):
    """
    Load data from Kaggle txt file.
    Format:
    ID ::: TITLE ::: GENRE ::: DESCRIPTION
    """
    
    data = []
    
    with open(file_path, "r", encoding="utf-8") as file:
        for line in file:
            parts = line.strip().split(" ::: ")
            
            if len(parts) == 4:
                _, title, genre, description = parts
                data.append([genre, description])
    
    df = pd.DataFrame(data, columns=["genre", "description"])
    return df