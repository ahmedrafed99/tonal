from kedro.pipeline import Pipeline, node
from .nodes import preprocess_data, split_dataset

def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            node(
                func=preprocess_data,
                inputs="raw_data",
                outputs="preprocessed_data",
                name="preprocess_data_node"
            ),
            node(
                func=split_dataset,
                inputs="preprocessed_data",
                outputs=["x_train", "x_test", "y_train", "y_test"],
                name="node_split_transform_daily_data"
            )
        ]
    )
