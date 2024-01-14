import numpy as np
import pandas as pd
import networkx as nx

from spektral.data.loader import SingleLoader

def merge_graph(file_prefix, file_list, minimum_contig_length, read_label=True):
    """
    Merge multiple graph files into main graph
    """
    # Read DataFrame
    train_file = pd.read_csv(file_list, names=('graph', 'csv', 'sample_id'))

    # Initialize an empty graph
    graph = nx.Graph()

    # Iterate through each row in the file list
    for _, row in train_file.iterrows():
        # Construct file path
        graph_file = file_prefix + row['graph']
        csv_file = file_prefix + row['csv'] if read_label else None

        # Read graph data to the main graph
        read_graph(graph_file, csv_file, row['sample_id'], graph, minimum_contig_length)

def apply_to_graph(model, graph, parameters, apply_thresholds=True):
    """
    Apply predictions from a machine learning model to a graph.

    Args:
    - model: Model used for prediction.
    - graph: Data graph to apply the model's predictions.
    - parameters: Dictionary containing parameters for predictions.
    - apply_thresholds: Boolean flag indicating whether to apply thresholds. Default is True.

    Returns:
    - preds: Predictions after processing.
    """

    # Create a SingleLoader instance for the graph
    loader = SingleLoader(graph)

    # Predict using the model on the loaded graph
    preds = model.predict(loader.load(), steps=loader.steps_per_epoch)

    # Normalize predictions based on the loss function used
    if parameters["loss_function"] == "squaredhinge":
        # Map predictions from the [-1, 1] scale to [0, 1]
        preds = (preds + 1) / 2.0

    # Clip predictions to ensure they are within the [0, 1] range
    preds = np.clip(preds, 0, 1)

    if apply_thresholds:
        # Apply thresholds to the predictions based on parameters.
        preds = apply_thresholds(preds, parameters)

        # Ensure predictions are within [0, 1] after applying thresholds
        preds = np.clip(preds, 0, 1)

    return preds