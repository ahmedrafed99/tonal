import pandas as pd
from sklearn.model_selection import train_test_split


def preprocess_data(input_datas: pd.DataFrame) -> pd.DataFrame:
    # Drop rows with missing or non-numeric values
    cleaned_data = input_datas.dropna()
    cleaned_data = cleaned_data.apply(pd.to_numeric, errors='coerce').dropna()

    # Classement des colonnes par ordre alphabétique
    shaped_datas = cleaned_data.sort_index(axis=1)
    normalized = normalize_csv_data(shaped_datas)
    return normalized


import pandas as pd
from sklearn.preprocessing import MinMaxScaler



def normalize_csv_data(input_data:pd.DataFrame) -> pd.DataFrame:
    """
    Normalize the data in a CSV file and save the normalized data to a new CSV file.

    Parameters:
    input_csv_path (str): Path to the input CSV file.
    output_csv_path (str): Path to save the normalized CSV file.
    """
    # Load the data into a pandas DataFrame

    df = input_data.apply(pd.to_numeric)

    # Normalize the data using MinMaxScaler
    scaler = MinMaxScaler()
    normalized_data = scaler.fit_transform(df)

    # Convert the normalized data back to a DataFrame
    normalized_df = pd.DataFrame(normalized_data, columns=df.columns)

    return normalized_df



def split_dataset(input_data: pd.DataFrame):

    X = input_data.filter(regex='^before_')
    y = input_data.filter(regex='^after_')

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test