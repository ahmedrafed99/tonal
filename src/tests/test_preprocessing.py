import pytest
import os
import sys
import pandas as pd
from unittest import mock


# Obtenez le répertoire racine du projet
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))

# Ajoutez le répertoire racine au chemin de recherche Python
sys.path.append(project_root)

# Now you can import modules from src without relative imports
#from tonal_project.pipelines.model_creation.nodes import create_model, train_model
from src.tonal_project.pipelines.model_creation.pipeline import create_pipeline
from src.tonal_project.pipelines.preprocessing.nodes import preprocess_data, split_dataset


@pytest.fixture
def sample_raw_data():
    return pd.DataFrame({
        "feature1": [1, 2, 3, 4, 5],
        "feature2": [10, 20, 30, 40, 50],
        "target": [100, 200, 300, 400, 500],
        "before_exam_125_Hz": [1, 2, 3, 4, 5],
        "before_exam_250_Hz": [1, 2, 3, 4, 5],
        "before_exam_500_Hz": [1, 2, 3, 4, 5],
        "before_exam_8000_Hz": [1, 2, 3, 4, 5],
        "after_exam_125_Hz": [1, 2, 3, 4, 5],
        "after_exam_250_Hz": [1, 2, 3, 4, 5],
        "after_exam_500_Hz": [1, 2, 3, 4, 5],
        "after_exam_8000_Hz": [1, 2, 3, 4, 5]
    })

@pytest.fixture
def shaped_data():
    return pd.DataFrame({
        "feature1": [1, 2, 3, 4, 5],
        "feature2": [10, 20, 30, 40, 50]
    }), pd.DataFrame({
        "target": [100, 200, 300, 400, 500]
    })

def test_preprocess_data(sample_raw_data):
    transformed_data = preprocess_data(sample_raw_data)
    assert transformed_data is not None
    assert isinstance(transformed_data, pd.DataFrame)

def test_split_dataset(shaped_data):
    data = pd.concat([shaped_data[0], shaped_data[1]], axis=1)
    x_train, x_test, y_train, y_test = split_dataset(data)
    assert len(x_train) > 0
    assert len(x_test) > 0
    assert len(y_train) > 0
    assert len(y_test) > 0

def test_create_pipeline():
    pipeline = create_pipeline()
    print(pipeline.nodes)  # Affiche les noms des nœuds dans le pipeline
    assert pipeline is not None
    assert len(pipeline.nodes) == 1  # Check the number of nodes in the pipeline
    node_names = [node.name for node in pipeline.nodes]
   

