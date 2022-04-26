from __future__ import print_function
import numpy as np
import math
import tensorflow as tf
from tensorflow.keras.layers import Dense, Convolution2D, Convolution1D, Convolution3D, Conv1D, Conv2D, Conv3D
from tensorflow.keras.models import Model

class WeightInitScheme1Params:
    def __init__(self, batch, use_gram_schmidt, verbose, active_frac, goal_std):
        self.batch = batch
        self.use_gram_schmidt = use_gram_schmidt
        self.verbose = verbose
        self.active_frac = active_frac
        self.goal_std = goal_std


class WeightInitScheme1:

    # @staticmethod
    # def initialize(model, params):
    #     return WeightInitScheme1.initialize_layers(model, model.layers, params)
    #
    #
    # @staticmethod
    # def initialize_layers(model, layer_lst, params):
    #     batch = params.batch
    #     active_frac = params.active_frac
    #     goal_std = params.goal_std
    #     use_gram_schmidt = params.use_gram_schmidt if params.use_gram_schmidt is not None else False
    #     verbose = params.verbose if params.verbose is not None else True
    #
    #     layers_initialized = 0
    #
    #     if verbose:
    #         print("------- Scheme 1 - Initialization Process Started ------- ")
    #
    #     for i in range(len(layer_lst)):
    #         layer = layer_lst[i]
    #
    #         # try:
    #         classes_to_consider = (Dense, Convolution2D, Convolution1D, Convolution3D, Conv1D, Conv2D, Conv3D)
    #
    #         if not isinstance(layer, classes_to_consider):
    #             if verbose:
    #                 print("Scheme1 - skipping " + layer.name + ' - not in the list of classes to be initialized')
    #             continue
    #
    #         weights_and_biases = layer.get_weights()
    #
    #         last_dim = weights_and_biases[0].shape[-1]
    #
    #         # Step 1 - Orthonormalization
    #         # Forcing weight tensor to be a 2D matrix, so we can make each column orthogonal to each other
    #         weights_reshaped = weights_and_biases[0].reshape((-1, last_dim))
    #         if use_gram_schmidt:
    #             weights_reshaped = Utils.gram_schmidt(weights_reshaped, False, False)
    #         else:
    #             weights_reshaped = Utils.svd_orthonormal(weights_reshaped.shape)
    #
    #         weights_and_biases[0] = np.reshape(weights_reshaped, weights_and_biases[0].shape)
    #         weights_and_biases[1] = np.zeros(weights_and_biases[1].shape)
    #
    #         layer.set_weights(weights_and_biases)
    #
    #         # Step 2 - ReLU Adaptation
    #         if active_frac is not None:
    #             # Get layer's activations before ReLU
    #             raw = Utils.get_layer_linear_activations(model, layer, batch)
    #
    #             raw = raw.reshape((-1, raw.shape[-1]))
    #
    #             # Sort all columns in the activation matrix
    #             r = raw.shape[0]
    #             sorted_raw = np.sort(raw, axis=0)
    #
    #             # The bias for each unit is set to the negative of the nth value in the activations for that unit,
    #             # where n is given by the active_frac hyper-parameter
    #             new_biases = -sorted_raw[math.floor(r - active_frac * r), :]
    #
    #             weights_and_biases[1] = new_biases
    #             layer.set_weights(weights_and_biases)
    #
    #         # Step 3 - Standarization
    #         if goal_std is not None:
    #             # Get layer's activations using the initialization set
    #             activations = Utils.get_layer_activations(model, layer, batch)
    #             activations = activations.reshape((-1, activations.shape[-1]))
    #
    #             h1_s_std = np.std(activations, axis=0)
    #
    #             weights_and_biases = layer.get_weights()
    #             last_dim = weights_and_biases[0].shape[-1]
    #             new_weights = weights_and_biases[0].reshape((-1, last_dim))
    #             new_biases = weights_and_biases[1]
    #
    #             # Compute new weights/biases - dive them by std of the activation and multiplying times desired std
    #             for j in range(new_weights.shape[1]):
    #                 new_weights[:, j] = new_weights[:, j] / h1_s_std[j] * goal_std
    #                 new_biases[j] = new_biases[j] / h1_s_std[j] * goal_std
    #
    #             new_weights = np.reshape(new_weights, weights_and_biases[0].shape)
    #             weights_and_biases[0] = new_weights
    #             weights_and_biases[1] = new_biases
    #
    #             layer.set_weights(weights_and_biases)
    #
    #         # Print some statistics about the weights/biases and the layer's activations
    #         if verbose:
    #             weights_and_biases = layer.get_weights()
    #
    #             new_weights = weights_and_biases[0].reshape((-1, last_dim))
    #             new_biases = weights_and_biases[1]
    #
    #             activations = Utils.get_layer_activations(model, layer, batch)
    #             activations = activations.reshape((-1))
    #
    #             new_weights = new_weights.reshape((-1, new_weights.shape[-1]))
    #             new_biases = new_biases.reshape((-1, new_biases.shape[-1]))
    #
    #             print("------- Scheme 1 - Layer initialized: " + layer.name + " ------- ")
    #
    #             print("Weights -- Std: ", np.std(new_weights), " Mean: ", np.mean(new_weights), " Max: ",
    #                   np.max(new_weights), " Min: ", np.min(new_weights))
    #
    #             print("Biases -- Std: ", np.std(new_biases), " Mean: ", np.mean(new_biases), " Max: ",
    #                   np.max(new_biases), " Min: ", np.min(new_biases))
    #
    #             print("Layer activations' std: ", np.std(activations, axis=0))
    #             print("Layer activations <= 0: ", (len(activations[activations <= 0]) / len(activations)))
    #             print("Layer activations >  0: ", (len(activations[activations > 0]) / len(activations)))
    #
    #         layers_initialized += 1
    #         # except Exception as ex:
    #         #
    #         #     print("Could not initialize layer: ", layer.name, " Error: ", ex)
    #         #     continue
    #
    #     if verbose:
    #         print("------- Scheme 1 - DONE - total layers initialized ", layers_initialized, "------- ")
    #
    #     return model

    @staticmethod
    def initialize_layers_effnet(model, layer_dict, params):
        batch = params.batch
        active_frac = params.active_frac
        goal_std = params.goal_std
        use_gram_schmidt = params.use_gram_schmidt if params.use_gram_schmidt is not None else False
        verbose = params.verbose if params.verbose is not None else True

        layers_initialized = 0

        if verbose:
            print("------- Scheme 1 - Initialization Process Started ------- ")

        for layer, parent in layer_dict.items():
            # try:
            classes_to_consider = (Dense, Convolution2D, Convolution1D, Convolution3D, Conv1D, Conv2D, Conv3D)

            if not isinstance(layer, classes_to_consider):
                if verbose:
                    print("Scheme1 - skipping " + layer.name + ' - not in the list of classes to be initialized')
                continue

            weights_and_biases = layer.get_weights()

            last_dim = weights_and_biases[0].shape[-1]

            # Step 1 - Orthonormalization
            # Forcing weight tensor to be a 2D matrix, so we can make each column orthogonal to each other
            weights_reshaped = weights_and_biases[0].reshape((-1, last_dim))
            if use_gram_schmidt:
                weights_reshaped = Utils.gram_schmidt(weights_reshaped, False, False)
            else:
                weights_reshaped = Utils.svd_orthonormal(weights_reshaped.shape)

            weights_and_biases[0] = np.reshape(weights_reshaped, weights_and_biases[0].shape)
            weights_and_biases[1] = np.zeros(weights_and_biases[1].shape)

            layer.set_weights(weights_and_biases)

            # Step 2 - ReLU Adaptation
            if active_frac is not None:
                # Get layer's activations before ReLU
                raw = Utils.get_layer_linear_activations_effnet(model, layer, parent, batch)

                raw = raw.reshape((-1, raw.shape[-1]))

                # Sort all columns in the activation matrix
                r = raw.shape[0]
                sorted_raw = np.sort(raw, axis=0)

                # The bias for each unit is set to the negative of the nth value in the activations for that unit,
                # where n is given by the active_frac hyper-parameter
                new_biases = -sorted_raw[math.floor(r - active_frac * r), :]

                weights_and_biases[1] = new_biases
                layer.set_weights(weights_and_biases)

            # Step 3 - Standarization
            if goal_std is not None:
                # Get layer's activations using the initialization set
                activations = Utils.get_layer_activations_effnet(model, layer, parent, batch)
                activations = activations.reshape((-1, activations.shape[-1])).astype('float32')

                h1_s_std = np.std(activations, axis=0)
                weights_and_biases = layer.get_weights()
                last_dim = weights_and_biases[0].shape[-1]
                new_weights = weights_and_biases[0].reshape((-1, last_dim))
                new_biases = weights_and_biases[1]

                # Compute new weights/biases - dive them by std of the activation and multiplying times desired std
                for j in range(new_weights.shape[1]):
                    new_weights[:, j] = new_weights[:, j] / h1_s_std[j] * goal_std
                    new_biases[j] = new_biases[j] / h1_s_std[j] * goal_std

                new_weights = np.reshape(new_weights, weights_and_biases[0].shape)
                weights_and_biases[0] = new_weights
                weights_and_biases[1] = new_biases

                layer.set_weights(weights_and_biases)

            # Print some statistics about the weights/biases and the layer's activations
            if verbose:
                weights_and_biases = layer.get_weights()

                new_weights = weights_and_biases[0].reshape((-1, last_dim))
                new_biases = weights_and_biases[1]

                activations = Utils.get_layer_activations_effnet(model, layer, parent, batch)

                activations = activations.reshape((-1)).astype('float32')

                new_weights = new_weights.reshape((-1, new_weights.shape[-1]))
                new_biases = new_biases.reshape((-1, new_biases.shape[-1]))

                print("------- Scheme 1 - Layer initialized: " + layer.name + " parent: " + parent.name +  " ------- ")

                print("Weights -- Std: ", np.std(new_weights), " Mean: ", np.mean(new_weights), " Max: ",
                      np.max(new_weights), " Min: ", np.min(new_weights))

                print("Biases -- Std: ", np.std(new_biases), " Mean: ", np.mean(new_biases), " Max: ",
                      np.max(new_biases), " Min: ", np.min(new_biases))

                print("Layer activations' std: ", np.std(activations, axis=0))
                print("Layer activations <= 0: ", (len(activations[activations <= 0]) / len(activations)))
                print("Layer activations >  0: ", (len(activations[activations > 0]) / len(activations)))

            layers_initialized += 1
            # except Exception as ex:
            #
            #     print("Could not initialize layer: ", layer.name, " Error: ", ex)
            #     continue

        if verbose:
            print("------- Scheme 1 - DONE - total layers initialized ", layers_initialized, "------- ")

        return model


class Utils:

    @staticmethod
    def create_init_set_kmeans(x_input_set, y_input_set, num_elements_in_init_set, random_seed):
        # Assumption: First dimension in input_set_x is the number of instances
        # Assumption: One-hot encoding used in input_set_y

        encoding_set = np.unique(y_input_set, axis=0)

        x_desired_shape = list(x_input_set.shape)
        x_desired_shape[0] = 0

        y_desired_shape = list(y_input_set.shape)
        y_desired_shape[0] = 0

        x_init_set = np.empty(x_desired_shape)
        y_init_set = np.empty(y_desired_shape)

        for i in range (encoding_set.shape[0]):
            instance_indexes = np.where((y_input_set == encoding_set[i]).all(axis=1))[0]
            x_sub_set = x_input_set[instance_indexes]

            fraction_instances = x_sub_set.shape[0] / x_input_set.shape[0]
            num_clusters = int(round(num_elements_in_init_set * fraction_instances))

            x_sub_set_reshaped = x_sub_set.reshape(x_sub_set.shape[0], -1)

            kmeans = KMeans(n_clusters=num_clusters, random_state=random_seed).fit(x_sub_set_reshaped)
            neigh = NearestNeighbors(n_neighbors=1)
            neigh.fit(x_sub_set_reshaped)

            centers = kmeans.cluster_centers_

            init_set_neighs_indexes = neigh.kneighbors(centers, n_neighbors=1, return_distance=False)
            init_set_neighs = x_sub_set_reshaped[init_set_neighs_indexes]

            desired_shape = list(x_sub_set.shape)
            desired_shape[0] = num_clusters

            x_init_set_encoding_i = init_set_neighs.reshape(desired_shape)
            y_init_set_encoding_i = np.tile(encoding_set[i], (num_clusters, 1))

            x_init_set = np.concatenate((x_init_set, x_init_set_encoding_i), axis=0)
            y_init_set = np.concatenate((y_init_set, y_init_set_encoding_i), axis=0)

        return x_init_set, y_init_set

    @staticmethod
    def gram_schmidt(X, row_vecs=True, norm=True):
        if not row_vecs:
            X = X.T
        Y = X[0:1, :].copy()
        for i in range(1, X.shape[0]):
            proj = np.diag((X[i, :].dot(Y.T) / np.linalg.norm(Y, axis=1) ** 2).flat).dot(Y)
            Y = np.vstack((Y, X[i, :] - proj.sum(0)))
        if norm:
            Y = np.diag(1 / np.linalg.norm(Y, axis=1)).dot(Y)
        if row_vecs:
            return Y
        else:
            return Y.T

    @staticmethod
    def svd_orthonormal(shape):
        # Orthonorm init code is taked from Lasagne
        # https://github.com/Lasagne/Lasagne/blob/master/lasagne/init.py
        if len(shape) < 2:
            raise RuntimeError("Only shapes of length 2 or more are supported.")
        flat_shape = (shape[0], np.prod(shape[1:]))
        a = np.random.standard_normal(flat_shape)
        u, _, v = np.linalg.svd(a, full_matrices=False)
        q = u if u.shape == flat_shape else v
        q = q.reshape(shape)
        return q

    @staticmethod
    def get_layer_linear_activations(model, layer, x_batch):
        r = model(x_batch, with_endpoints=True)
        print(model.endpoints)
        org_act = layer.activation
        layer.activation = tf.keras.activations.linear

        print(model.input)
        print(layer.output)

        # inter_model = Model(inputs=model.input, outputs=layer.output)
        inter_model = Model(model.input, layer.output, model.output)

        activations = inter_model.predict(x_batch)

        layer.activation = org_act
        return activations

    @staticmethod
    def get_layer_linear_activations_effnet(model, layer, parent, x_batch):
        org_act = layer.activation
        layer.activation = tf.keras.activations.linear
        model(x_batch)

        activations = parent.activations[layer.name].numpy()

        layer.activation = org_act
        return activations

    @staticmethod
    def get_layer_activations(model, layer, x_batch):
        inter_model = Model(inputs=model.input, outputs=layer.output)

        activations = inter_model.predict(x_batch)
        return activations

    @staticmethod
    def get_layer_activations_effnet(model, layer, parent, x_batch):
        model(x_batch)
        activations = parent.activations[layer.name+"_after_act"].numpy()

        return activations