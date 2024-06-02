import pytest
import os
import sys
import pandas as pd
from unittest import mock
from keras import Model

# Obtenez le répertoire racine du projet
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))

# Ajoutez le répertoire racine au chemin de recherche Python
sys.path.append(project_root)

# Now you can import modules from src without relative imports
from src.tonal_project.pipelines.model_creation.nodes import create_model, train_model
from src.tonal_project.pipelines.model_creation.pipeline import create_pipeline

@pytest.fixture
def sample_data():
    return pd.DataFrame({
        "feature1": [1, 2, 3, 4, 5],
        "feature2": [10, 20, 30, 40, 50]
    }), pd.DataFrame({
        "target": [100, 200, 300, 400, 500]
    })

def test_create_model():
    input_shape = (10, 5, 1)
    model = create_model(input_shape)
    assert isinstance(model, Model)

def test_train_model(mocker, sample_data):
    X, y = sample_data
    mock_model = mocker.Mock()
    mock_create_model = mocker.patch("src.tonal_project.pipelines.model_creation.nodes.create_model", return_value=mock_model)
    mock_early_stopping = mocker.patch("keras.callbacks.EarlyStopping", return_value=mocker.Mock())  # Patch keras.callbacks.EarlyStopping

    x_train, x_test = X.iloc[:3], X.iloc[3:]
    y_train, y_test = y.iloc[:3], y.iloc[3:]

    model = train_model(x_train, y_train, x_test, y_test)

    # Verify if the model was created and trained
    mock_create_model.assert_called_once()
    mock_model.fit.assert_called_once()
    assert model == mock_model

def test_create_pipeline():
    pipeline = create_pipeline()
    assert pipeline is not None
    assert len(pipeline.nodes) == 1  # Check the number of nodes in the pipeline

    node_names = [node.name for node in pipeline.nodes]
    assert "node_train_model" in node_names

def test_create_model_invalid_shape():
    invalid_input_shape = (10, 5)  # Only 2 dimensions instead of 3
    with pytest.raises(ValueError, match="Expected input shape to have 3 dimensions"):
        create_model(invalid_input_shape)
