"""Module for changing setting optimal threshold wrt F1 measure on validation set and applying them to new data"""

import os
import numpy as np

from utils.graphs import apply_to_graph


def set_thresholds(model, graph, weights, parameters, log_dir=None):
    """Set thresholds during training, store in parameters"""

    # Get predictions using the model and graph
    y_pred = apply_to_graph(model, graph, parameters, apply_thresholds=False)

    # Extract true labels from the graph
    y_true = graph.extract_y()

    # Compute scores and thresholds for plasmids
    plasmid_scores = score_thresholds(y_true[:, 0], y_pred[:, 0], weights)
    store_best(plasmid_scores, parameters, 'plasmid_threshold', log_dir)

    # Compute scores and thresholds for chromosomes
    chromosome_scores = score_thresholds(y_true[:, 1], y_pred[:, 1], weights)
    store_best(chromosome_scores, parameters, 'chromosome_threshold', log_dir)

def apply_thresholds(y, parameters):
    """Apply thresholds during testing, return transformed scores so that 0.5 corresponds to threshold"""

    columns = []  # List to store transformed columns

    # Loop through each column (plasmids and chromosomes)
    for (column_idx, which_parameter) in [(0, 'plasmid_threshold'), (1, 'chromosome_threshold')]:
        # Extract threshold from parameters
        threshold = parameters[which_parameter]
        
        # Extract the original column values
        original_column = y[:, column_idx]

        # Apply the scaling function with different parameters for small and large numbers
        new_column = np.piecewise(
            original_column,
            [original_column < threshold, original_column >= threshold],
            [lambda x: scale_number(x, 0, threshold, 0, 0.5), lambda x: scale_number(x, threshold, 1, 0.5, 1)]
        )

        # Add the transformed column to the list
        columns.append(new_column)

    # Transpose the list of columns to create the final transformed array
    y_new = np.array(columns).transpose()
    
    return y_new

def scale_number(x, in_min, in_max, out_min, out_max):
    """
    Scale the input number 'x' from the range [in_min, in_max] to the range [out_min, out_max].
    """
    # Calculate the scaled value using linear interpolation
    return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min

def store_best(scores, parameters, which, log_dir):
    """Store the optimal threshold for one output in parameters and, if requested, print all thresholds to a log file"""

    # Check if the scores list is not empty
    if len(scores) > 0:
        # Find the index of the maximum F1 score in the scores list
        maxindex = max(range(len(scores)), key=lambda i: scores[i][1])
        # Retrieve the corresponding threshold using the found index
        threshold = scores[maxindex][0]
    else:
        # If the input array is empty, use the default threshold of 0.5
        threshold = 0.5

    # Store the found threshold in the parameters dictionary
    parameters[which] = float(threshold)

    # Check if log_dir is provided
    if log_dir is not None:
        # Construct the filename based on the output 'which'
        filename = os.path.join(log_dir, which + ".csv")

        # Open the file and write thresholds and F1 scores
        with open(filename, 'wt') as file:
            # Write column headers to the file
            print(f"{which},F1", file=file)

            # Write each pair of threshold and F1 score to the file
            for score in scores:
                print(",".join(str(value) for value in score), file=file)

def score_thresholds(y_true, y_pred, weights):
    """Compute F1 score of all thresholds for one output (plasmid or chromosome)"""

    # Ensure that the shapes of y_true, y_pred, and weights are valid
    assert y_true.shape == y_pred.shape == weights.shape, "Input shapes mismatch"

    # Get data points with non-zero weight and create a list of pairs
    pairs = [(y_true[i], y_pred[i]) for i in range(y_true.shape[0]) if weights[i] > 0]

    # Sort the pairs based on predicted values in descending order
    pairs.sort(key=lambda x: x[1], reverse=True)

    # Count all positives in true labels
    positives = sum(1 for pair in pairs if pair[0] > 0.5)

    scores = []
    tp = 0  # Initialize true positives

    # Iterate through the sorted pairs to compute F1 scores
    for i in range(len(pairs)):
        # Increase true positives if true label is above threshold
        if pairs[i][0] > 0.5:
            tp += 1

        # Calculate F1 score when the predicted value changes
        if i > 0 and pairs[i][1] < pairs[i-1][1]:
            recall = tp / positives
            precision = tp / (i + 1)
            f1 = 2 * precision * recall / (precision + recall)
            threshold = (pairs[i-1][1] + pairs[i][1]) / 2
            scores.append((threshold, f1))

    return scores