import yaml

from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

from spektral.transforms import GCNFilter
from spektral.data.loaders import SingleLoader

from models import plASgraph

def train(config_file, file_prefix, file_list):
    with open(config_file, 'r') as config:
        parameters = yaml.safe_load(config)

    random_seed = parameters['random_seed']

    # Return a NetworkX graph object containing the merged graph data.
    G = merge_graph(file_prefix, file_list, parameters['minimum_contig_length'])

    nodelist = list(G)

    # Convert from NetworkX graph to Spektral graph
    graphs = NetworkX_to_Spektral(G, nodelist, parameters)

    # Normalize by degrees, and 1 along diagonal
    graphs.apply(GCNFilter())

    model = plASgraph(parameters=parameters)

    if parameters['loss_function'] == 'squaredhinge':
        loss_function = tf.keras.losses.SquaredHInge(reduction = 'sum')
    if parameters['loss_function'] == 'crossentropy':
        loss_function = tf.keras.losses.BinaryCrossentropy(reduction = 'sum')
    if parameters['loss_function'] == 'mse':
        loss_function = tf.keras.losses.MeanSquaredError(reduction = 'sum')
    else:
        raise ValueError(f"Loss function error")

    print(loss_function)

    model.compile(
        optimizer = Adam(parameters['learning_rate']),
        loss_function = loss_function,
        weighted_metrics = []
    )

    label_weights = {
        "unlabeled": 0,
        "chromosome": 1,
        "plasmid": parameters["plasmid_ambiguous_weight"],
        "ambiguous": parameters["plasmid_ambiguous_weight"]
    }

    # Sample's weight and masking
    # labels = [graph.nodes[node_id]['text_label'] for node_id in nodelist]
    mask = [label_weights.get(G.nodes[node_id]['text_label'], 0) for node_id in nodelist]

    masks_train = mask.copy()
    masks_val = mask.copy()

    masks_train = np.array(masks_train).astype(float)
    masks_val = np.array(masks_val).astype(float)

    train_loader = SingleLoader(graphs, sample_weights = masks_train)
    val_loader = SingleLoader(graphs, sample_weights = masks_val)

    history = model.fit(
        train_loader.load(),
        validation_data = val_loader.load(),
        steps_per_epoch = train_loader.steps_per_epoch,
        validation_steps = val_loader.steps_per_epoch,
        epochs = parameters['epochs'],
        callbacks = [
            EarlyStopping(
                patience = parameters['early_stopping_patience'],
                monitor = 'val_loss',
                mode = 'min',
                verbose = 1,
                restore_best_weights = True
            )
        ]
    )

    # if parameters['set_thresholds']:
    #   thresholds.set_thresholds(model, graphs, mask_val, parameters)

    # model.save(model_output_dir)