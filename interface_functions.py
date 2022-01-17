import struct
import numpy as np


class NeuralNetworkDecoder:

    def __init__(self, architecture):
        architecture = architecture.astype(int)
        dims_per_layer = 3
        self.nb_dense_layers = len(architecture) // dims_per_layer

        self.weight_shapes = []
        for _i in range(self.nb_dense_layers):
            self.weight_shapes.append((architecture[_i * dims_per_layer], architecture[_i * dims_per_layer + 1]))
            self.weight_shapes.append((architecture[_i * dims_per_layer + 2],))

        self.message_lens = []

    def recv_first_network(self, socket):

        model_weights = []
        weights_array = None
        for _i in self.weight_shapes:
            nb_elements = np.prod(_i)
            while weights_array is None or len(weights_array) < nb_elements:
                binary_weights = socket.recv(1024)
                self.message_lens.append(len(binary_weights))
                if weights_array is None:
                    weights_array = np.frombuffer(binary_weights, dtype=np.float32)
                else:
                    weights_array = np.append(weights_array, np.frombuffer(binary_weights, dtype=np.float32))

            if len(weights_array) > nb_elements:
                model_weights.append(np.reshape(weights_array[:nb_elements], _i))
                weights_array = weights_array[nb_elements:]
            else:
                model_weights.append(np.reshape(weights_array, _i))
                weights_array = None

        _layer_message = []
        grouped_message_lens = []
        self.message_lens
        self.message_len = np.sum(self.message_lens)
        # for message_len in self.message_lens:
        #     if len(_layer_message) == 0 or _layer_message[-1] == 1024:
        #         _layer_message.append(message_len)
        #     else:
        #         grouped_message_lens.append(_layer_message)
        #         _layer_message = []
        #         _layer_message.append(message_len)
        # grouped_message_lens.append(_layer_message)
        # self.message_lens = grouped_message_lens
        self.message_lens = [1024] * (self.message_len // 1024)
        self.message_lens.append(self.message_len % 1024)

        return model_weights

    def recv_network(self, socket):

        model_weights = []
        weights_array = None
        remaining_message_len = np.copy(self.message_len)
        for _i in self.weight_shapes:
            nb_elements = np.prod(_i)
            while weights_array is None or len(weights_array) < nb_elements:
                next_snippet_len = 1024 if remaining_message_len >= 1024 else remaining_message_len
                binary_weights = socket.recv(next_snippet_len)
                remaining_message_len -= len(binary_weights)
                if weights_array is None:
                    weights_array = np.frombuffer(binary_weights, dtype=np.float32)
                else:
                    weights_array = np.append(weights_array, np.frombuffer(binary_weights, dtype=np.float32))

            if len(weights_array) > nb_elements:
                model_weights.append(np.reshape(weights_array[:nb_elements], _i))
                weights_array = weights_array[nb_elements:]
            else:
                model_weights.append(np.reshape(weights_array, _i))
                weights_array = None

        return model_weights

