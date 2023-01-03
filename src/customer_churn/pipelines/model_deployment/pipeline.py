"""
This is a boilerplate pipeline 'model_deployment'
generated using Kedro 0.18.3
"""

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import data_oot, split_xy_oot, predict_proba

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func = data_oot,
            inputs = ["data_oot", "features"],
            outputs = ["data_gojek", "data_grab"],
            name = "data_oot_node"
        ),
        node(
            func = split_xy_oot,
            inputs = "data_gojek",
            outputs = ["X_gojek", "y_gojek"],
            name = "split_xy_gojek_node"
        ),
        node(
            func = split_xy_oot,
            inputs = "data_grab",
            outputs = ["X_grab", "y_grab"],
            name = "split_xy_grab_node"
        ),
        node(
            func = predict_proba,
            inputs = ["X_gojek", "y_gojek", "model_cat"],
            outputs = "gojek_prediction",
            name = "predict_proba_gojek_node"
        ),
        node(
            func = predict_proba,
            inputs = ["X_grab", "y_grab", "model_cat"],
            outputs = "grab_prediction",
            name = "predict_proba_grab_node"
        )
    ])
