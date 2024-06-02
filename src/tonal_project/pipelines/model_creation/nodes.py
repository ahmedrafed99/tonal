from keras import layers, regularizers, optimizers, metrics, Model
import pandas as pd
import mlflow

mlflow.autolog()

def create_model(input_shape, units=128, activation='relu', l2_value=0.01, dropout_rate=None, learning_rate=1e-3):
    # Check if the input shape has 3 dimensions
    if len(input_shape) != 3:
        raise ValueError("Expected input shape to have 3 dimensions")

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

    return model
