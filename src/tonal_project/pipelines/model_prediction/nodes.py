from keras import Model
import pandas as pd
import numpy as np
from ..preprocessing.nodes import preprocess_data


def transform_data(input_data: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the data by ensuring all values are floats and removing any rows with null values.
    Keep only the necessary columns for prediction.
    """

    prediction_columns = ['before_exam_125_Hz', 'before_exam_250_Hz', 'before_exam_500_Hz',
                          'before_exam_1000_Hz', 'before_exam_2000_Hz', 'before_exam_4000_Hz',
                          'before_exam_8000_Hz']

    if not isinstance(input_data, pd.DataFrame):
        raise TypeError("Input data must be a pandas DataFrame")

    missing_columns = [
        col for col in prediction_columns if col not in input_data.columns]
    if missing_columns:
        raise ValueError(
            f"The following required columns are missing from the input data: {', '.join(missing_columns)}")

    data_transformed = input_data.apply(pd.to_numeric, errors='coerce')
    data_transformed.dropna(inplace=True)

    data_transformed = preprocess_data(data_transformed)
    data_transformed = data_transformed[prediction_columns]

    return data_transformed


def predict_model(input_data: pd.DataFrame, model: Model) -> pd.DataFrame:
    """
    Predict the output of the model using the input data.
    """

    data_transformed = transform_data(input_data)

    # Reshape the data to match the model's expected input shape
    data_reshaped = data_transformed.values.reshape((data_transformed.shape[0], data_transformed.shape[1], 1))

    # Reshape the data to match the model's expected input shape
    print(f"Transformed data shape: {data_transformed.shape}")
    data_predicted = model.predict(data_reshaped)
    print(f"Predicted data: {data_predicted}")

    # Adjust the column names to match the expected output
    predicted_columns = ['after_exam_125_Hz', 'after_exam_250_Hz', 'after_exam_500_Hz',
                         'after_exam_1000_Hz', 'after_exam_2000_Hz', 'after_exam_4000_Hz',
                         'after_exam_8000_Hz']

    df = pd.DataFrame(data_predicted, columns=predicted_columns)

    df = df.round().astype(int)

    return df
