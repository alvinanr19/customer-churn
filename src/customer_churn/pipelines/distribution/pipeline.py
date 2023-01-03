"""
This is a boilerplate pipeline 'distribution'
generated using Kedro 0.18.3
"""

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import distribution

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func = distribution,
            inputs = "training_results_cat",
            outputs = ["training_results_cat_final", "training_distribution"],
            name = "training_distribution_node"
        ),
        node(
            func = distribution,
            inputs = "testing_results_cat",
            outputs = ["testing_results_cat_final", "testing_distribution"],
            name = "testing_distribution_node"
        ),
        node(
            func = distribution,
            inputs = "gojek_prediction",
            outputs = ["gojek_prediction_final", "gojek_distribution"],
            name = "gojek_distribution_node"
        ),
        node(
            func = distribution,
            inputs = "grab_prediction",
            outputs = ["grab_prediction_final", "grab_distribution"],
            name = "grab_distribution_node"
        )
    ])
