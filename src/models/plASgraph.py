import tensorflow as tf

from tensorflow.keras import layers
from tensorflow.python.trackable.data_structures import NoDependency

class plASgraph2(tf.keras.Model):
    def __init__(self, parameters, **kwargs):
        super().__init__(**kwargs)

        self.parameters = parameters

        self.regularizers = tf.keras.regularizers.L2(self.parameters['L2_regularizers'])

        # Input layers
        self.preprocessing = layers.Dense(self.parameters['num_channels_preprocessing'], activations=self.parameters['preprocessing_activation'])
        self._fully_connected_input_1 = layers.Dense(self.parameters['num_channels_per_layer'], activation=self.parameters['fully_connected_activation'])
        self._fully_connected_input_2 = layers.Dense(self.parameters['num_channels_per_layer'], activation=self.parameters['fully_connected_activation'])

        # Graph Neural Networks iterations layers
        self._gnn_dropout_before_gcn = NoDependency([layers.Dropout(self.parameters['dropout_rate']) for _ in range(self.parameters['num_gnn_layers'])])
        self._gnn_dropout_before_fully_connected = NoDependency([layers.Dropout(self.parameters['dropout_rate']) for _ in range(self.parameters['num_gnn_layers'])])

        gcn_layers = [] # Temporary list of GCN layers
        fully_connected_layers = [] # Temporary list of fully connected layers

        # Iterate through the number of GNN layers and create names for instance variables
        for layer_index in range(self.parameters['num_gnn_layers']):
            # Create names of instance variables for the layers
            gcn_instance_name = f"gnn_gcn_{layer_index}"

            # Create instance variable names for fully connected layers
            dense_instance_name = f"gnn_dense_{layer_index}"

            # Add layers to lists
            gcn_layers.append(getattr(self, gcn_instance_name))
            fully_connected_layers.append(getattr(self, dense_instance_name))

        # Store list in instance variables which are not saved
        self._gnn_gcn = NoDependency(gcn_layers)
        self._gnn_fully_connected = NoDependency(fully_connected_layers)

        # Output layers
        self._fully_connected_output_1 = layers.Dense(self.parameters['num_channels_per_layer'], activation=self.parameters['fully_connected_activation'])
        self._fully_connected_output_2 = layers.Dense(self.parameters['num_channels_per_layer'], activation=self.parameters['fully_connected_activation'])
        self._dropout_output_1 = layers.Dropout(self.parameters['dropout_rate'])
        self._dropout_output_2 = layers.Dropout(self.parameters['dropout_rate'])


    def __getitem__(self, key):
        return self.parameters[key]

    def call(self, inputs):
        x, _ = inputs

        # Input layers
        x = self.preprocessing(x)
        node_identity = self._fully_connected_input_1(x)
        x = self._fully_connected_input_2(x)

        # Graph Neural Networks layers
        for gnn_layer in range(self.parameters['num_gnn_layers']):
            x = self._gnn_dropout_before_gcn[gnn_layer](x)
            x = self._gnn_gcn(gnn_layer)(x)

            merged = layers.concatenate([node_identity, x])

            x = self._gnn_dropout_before_fully_connected[gnn_layer](merged)
            x = self._gnn_fully_connected[gnn_layer](x)

        # Output layers
        merged = layers.concatenate([node_identity, x])

        x = self._dropout_output_1(merged)
        x = self._fully_connected_output_1(x)
        x = self._dropout_output_2(x)
        x = self._fully_connected_output_2(x)

        return x