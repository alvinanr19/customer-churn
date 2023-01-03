"""
This is a boilerplate pipeline 'model_development'
generated using Kedro 0.18.3
"""

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import split_data, bootstrap, split_xy, model_cat

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func = split_data,
            inputs = "preprocessed_dataset",
            outputs = ["training", "testing"],
            name = "split_dataset_node"
        ),
        node(
            func = bootstrap,
            inputs = "training",
            outputs = "bootstrapped_data_train",
            name = "bootstrap_node"
        ),
        node(
            func = split_xy,
            inputs = ["bootstrapped_data_train", "testing"],
            outputs = ["data_train", "data_test", "label_train", "label_test"],
            name = "split_xy_node"
        ),
        node(
            func = model_cat,
            inputs = ["data_train", "data_test", "label_train", "label_test"],
            outputs = ["model_cat", "training_results_cat", "testing_results_cat"],
            name = "model_cat_node"
        )
    ])
