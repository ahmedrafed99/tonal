import pandas as pd
from sklearn.model_selection import train_test_split
def preprocess_data(input_datas: pd.DataFrame) -> pd.DataFrame:
    # Drop rows with missing or non-numeric values
    cleaned_data = input_datas.dropna()
    cleaned_data = cleaned_data.apply(pd.to_numeric, errors='coerce').dropna()

    # Classement des colonnes par ordre alphabétique
    shaped_datas = cleaned_data.sort_index(axis=1)

    return shaped_datas

def split_dataset(input_data: pd.DataFrame):

    X = input_data.filter(regex='^before_')
    y = input_data.filter(regex='^after_')

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test