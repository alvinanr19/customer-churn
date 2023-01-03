"""
This is a boilerplate pipeline 'data_preprocessing'
generated using Kedro 0.18.3
"""

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import drop_column, mrmr_selection

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func = drop_column,
            inputs = "dataset",
            outputs = "dropped_dataset",
            name = "drop_dataset_node"
        ),
        node(
            func = mrmr_selection,
            inputs = "dropped_dataset",
            outputs = ["preprocessed_dataset", "features"],
            name = "mrmr_selection_node"
        )
    ])
