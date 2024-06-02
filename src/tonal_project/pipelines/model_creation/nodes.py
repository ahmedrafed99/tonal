"""
This is a boilerplate pipeline 'model_creation'
generated using Kedro 0.19.5
"""
from keras import layers, regularizers, optimizers, metrics, Model
import pandas as pd
import mlflow

import mlflow
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes
from sklearn.ensemble import RandomForestRegressor

mlflow.autolog()
db = load_diabetes()
X_train, X_test, y_train, y_test = train_test_split(db.data, db.target)
# Create and train models.
rf = RandomForestRegressor(n_estimators=100, max_depth=6, max_features=3)
rf.fit(X_train, y_train)
# Use the model to make predictions on the test dataset.
predictions = rf.predict(X_test)


def create_model(input_shape, units=128, activation='relu', l2_value=0.01, dropout_rate=None, learning_rate=1e-3):
    inputs = layers.Input(shape=(input_shape[1], 1))

    x = layers.Conv1D(filters=32, kernel_size=3, activation=activation, padding='same')(inputs)
    x = layers.MaxPooling1D(pool_size=2, padding='same')(x)
    x = layers.Conv1D(filters=64, kernel_size=3, activation=activation, padding='same')(x)
    x = layers.MaxPooling1D(pool_size=2, padding='same')(x)

    x = layers.Flatten()(x)

    x = layers.Dense(units, activation='relu', kernel_regularizer=regularizers.L2(l2_value))(x)

    if dropout_rate is not None:
        x = layers.Dropout(dropout_rate)(x)

    x = layers.Dense(input_shape[1], activation='softmax')(x)

    model = Model(inputs=inputs, outputs=x)

    model.compile(optimizer=optimizers.Adam(learning_rate=learning_rate),
                  loss="categorical_crossentropy",
                  metrics=[metrics.CategoricalAccuracy(), metrics.MeanAbsoluteError(), metrics.MeanSquaredError()])

    return model


def train_model(X_train: pd.DataFrame, y_train: pd.DataFrame, X_test: pd.DataFrame, y_test: pd.DataFrame):
    model = create_model(X_train.shape, dropout_rate=0.5)

    model.fit(X_train, y_train, epochs=50, validation_data=(X_test, y_test))
    # Save the model in .h5 format
    model.save("data/06_models/model_trained.h5")
    return model
