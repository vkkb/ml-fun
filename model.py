"""Feedforward neural network.

Feedforward neural network implemented with NumPy.
Used to review the mathematical background and the various steps involved
in a neural network.
"""

__author__ = 'Victor Kovacs'

import numpy as np


class NN(object):
    """Neural network"""

    def __init__(self):
        self.layers_count = 0
        self.layers_dict = {}
        self.history = {'epoch': [], 'error_train': [], 'error_test': []}

    def add_layer(self, units=1, activation=np.tanh,
                  initialization=np.random.uniform):
        """Updates dictionary with layer information"""

        self.layers_count += 1
        dict_entry = {'units': units, 'activation': activation,
                      'initialization': initialization}
        self.layers_dict[self.layers_count] = dict_entry

    def train(self, inputt, target, learn_rate=0.001, iterations=10001,
              history=100, inputt_test=[], target_test=[]):
        """Train neural network"""

        self._initialize_weights(inputt)
        for i in range(0, iterations):
            self._forwardprop(inputt)
            self._backprop(inputt, target)
            if (history and (i % history) == 0):
                self._save_history(i, target, inputt_test, target_test)
            # Update weights in each layer
            for layer in range(1, self.layers_count + 1):
                delta = np.multiply(learn_rate,
                                    self.layers_dict[layer]['partial_e_wrt_w'])
                delta *= (-1)
                self.layers_dict[layer]['weights'] += delta

    def predict(self, inputt):
        """Predict new outputs for the given input"""

        self._forwardprop(inputt, train=False)
        return self.prediction

    def sigmoid(self, x):
        """Logistic sigmoid"""

        return (1 / (1+np.exp(-x)))

    def sum_of_sqrs(self, t, pred):
        """Calculate sum of squares between predicted output and target t"""

        return(0.5 * np.power(np.sum(t - pred), 2))

    def _initialize_weights(self, inputt):
        """Initialize all weights of each unit in each layer

        Each hidden layer and output layer has a numpy array with two
        dimensions: (unit, feature), which are saved in a dictionary.
        """

        dim = inputt.shape[1] + 1  # +1 for the bias
        for layer in range(1, self.layers_count+1):
            weights = []
            layer_dict = self.layers_dict[layer]
            initializer = layer_dict['initialization']
            for unit in range(1, self.layers_dict[layer]['units'] + 1):  # Bias
                weights.append(initializer(size=dim))
            self.layers_dict[layer]['weights'] = np.array(weights)
            dim = layer_dict['units'] + 1

    def _forwardprop(self, inputt, train=True):
        """Propagate input forward through network

        The activations of each layer are stacked into matrices, where each
        column represents a unit + bias, and each row a observation.
        """

        # Include bias as additional column with ones in the input matrix
        inputt = np.hstack((np.ones(inputt.shape[0])[:, np.newaxis], inputt))

        # Iterate over all layers multiplying weights with activations
        for layer in range(1, self.layers_count + 1):
            if layer > 1:
                inputt = z_matrix
            units = self.layers_dict[layer]['units']
            a_matrix = np.zeros((inputt.shape[0], units+1))
            for unit in range(1, units + 2):
                if unit == 1:
                    a_matrix[:, unit-1] = np.ones(inputt.shape[0])
                else:
                    w = self.layers_dict[layer]['weights'][unit - 2]
                    w = w[:, np.newaxis]
                    a = np.matmul(inputt, w)
                    a_matrix[:, unit-1] = a.flatten()
            activation = self.layers_dict[layer]['activation']
            z_matrix = activation(a_matrix)

        # Save values that are necessary for backpropagation when training
            if train:
                self.layers_dict[layer]['a_matrix'] = a_matrix
                self.layers_dict[layer]['z_matrix'] = z_matrix
                if layer == self.layers_count:  # save output layer separately
                    self.output = z_matrix[:, 1][:, np.newaxis]
            else:
                if layer == self.layers_count:
                    self.prediction = z_matrix[:, 1][:, np.newaxis]

    def _backprop(self, inputt, t):
        """Backpropagation to update weights

        Assuming a sum of squares error function = 1/2 sum (y-t)^2
        """

        # calculate partial_net / partial_w_ij = z_i
        for layer in range(self.layers_count, 0, -1):
            if layer > 1:
                z_prev_layer = self.layers_dict[layer - 1]['z_matrix']
            else:
                z_prev_layer = inputt
                bias = np.ones(inputt.shape[0])[:, np.newaxis]
                z_prev_layer = np.hstack((bias, z_prev_layer))
            partial_net_wrt_w = z_prev_layer
            self.layers_dict[layer]['partial_net_wrt_w'] = partial_net_wrt_w

        # calculate partial_z_j / partial_net
            z_layer = self.layers_dict[layer]['z_matrix']
            if layer == self.layers_count:
                z_layer = z_layer[:, 1][:, np.newaxis]
            if self.layers_dict[layer]['activation'] == self.sigmoid:
                partial_z_wrt_net = np.multiply(z_layer, 1-z_layer)
            elif self.layers_dict[layer]['activation'] == np.tanh:
                partial_z_wrt_net = 1 - np.power(z_layer, 2)

            # Remove bias column since not affected by previous layer
            if layer < self.layers_count:
                partial_z_wrt_net = partial_z_wrt_net[:, 1:]
            self.layers_dict[layer]['partial_z_wrt_net'] = partial_z_wrt_net

        # Calculate partial_E / partial_z
        # Careful not to consider bias of subsequent layer in the sum
            if layer == self.layers_count:
                partial_e_wrt_z = self.output - t

            else:
                partial_e_wrt_z_l = self.layers_dict[layer + 1]['partial_e_wrt_z']
                partial_z_wrt_net_l = self.layers_dict[layer + 1]['partial_z_wrt_net']
                # Exclude bias weight of next layer, since no connection.
                weights_j_l = self.layers_dict[layer + 1]['weights'][:, 1:]
                partial_e_wrt_z = np.multiply(partial_e_wrt_z_l,
                                              partial_z_wrt_net_l)
                partial_e_wrt_z = np.matmul(partial_e_wrt_z, weights_j_l)

            delta_j = np.multiply(partial_e_wrt_z, partial_z_wrt_net)
            o_i = partial_net_wrt_w
            partial_e_wrt_w = np.matmul(delta_j.T, o_i)
            self.layers_dict[layer]['partial_e_wrt_z'] = partial_e_wrt_z
            self.layers_dict[layer]['partial_e_wrt_w'] = partial_e_wrt_w

    def _save_history(self, epoch, target, inputt_test, target_test):
        """Save train and test error during training"""

        self.history['epoch'].append(epoch)
        self.history['error_train'].append(self.sum_of_sqrs(target,
                                                            self.output))
        if len(inputt_test) > 0 and len(target_test) > 0:
            pred_test = self.predict(inputt_test)
            self.history['error_test'].append(self.sum_of_sqrs(target_test,
                                                               pred_test))
