import struct
import time

import numpy as np
import h5py
from System import Array
from pathlib import Path

class NeuralNetworkDecoder:

    def __init__(self, experiment_name, architecture):
        architecture = architecture.astype(int)
        dims_per_layer = 3
        self.nb_dense_layers = len(architecture) // dims_per_layer

        self.weight_shapes = []
        for _i in range(self.nb_dense_layers):
            self.weight_shapes.append((architecture[_i * dims_per_layer], architecture[_i * dims_per_layer + 1]))
            self.weight_shapes.append((architecture[_i * dims_per_layer + 2],))

        self.nb_neurons_per_layer = self.weight_shapes[0][1]
        self.message_lens = []#

        self.experiment_name = experiment_name

        self.pipeline_active = False
        self.save_network = False

        self.checkpoint_nb = 0
        self.started_saving = False

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

        self.message_len = np.sum(self.message_lens)

        self.message_lens = [1024] * (self.message_len // 1024)
        self.message_lens.append(self.message_len % 1024)

        return model_weights

    def network_acquisition(self, socket, MAPort, nn_parameter_paths, updateTime, ValueFactory):

        while True:
            if self.pipeline_active:
                break
            else:
                time.sleep(0.2)

        while self.pipeline_active:
            start = time.time()
            self.recv_network(socket)
            self.apply_network_FPGA(socket, MAPort, nn_parameter_paths, ValueFactory)
            Update_Time = time.time()-start

            MAPort.Write(
                 updateTime,
                 ValueFactory.CreateFloatValue(Update_Time)
            )

        self.save_network_weights()

        return None

    def recv_network(self, socket):

        self.model_weights = []
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
                self.model_weights.append(np.reshape(weights_array[:nb_elements], _i))
                weights_array = weights_array[nb_elements:]
            else:
                self.model_weights.append(np.reshape(weights_array, _i))
                weights_array = None

    def apply_network_FPGA(self, socket, MAPort, nn_parameter_paths, ValueFactory):
        socket.send(bytes(1))

        for _i in range(self.nb_dense_layers):
            w = np.transpose(np.append(self.model_weights[_i * 2], [self.model_weights[_i * 2 + 1]], axis=0))
            w_shape = np.shape(w)
            if _i == 0:
                w = np.append(w, np.zeros([w_shape[0], self.nb_neurons_per_layer + 1 - w_shape[1]]), axis=1)
            else:
                w = np.append(w, np.zeros([self.nb_neurons_per_layer - w_shape[0], w_shape[1]]), axis=0)

            # if _i == self.nb_dense_layers - 1:
            #     w = np.append(w, np.zeros([self.nb_neurons_per_layer - w_shape[0], w_shape[1]]), axis=0)
            # else:
            #     w = np.append(w, np.zeros([w_shape[0], self.nb_neurons_per_layer + 1 - w_shape[1]]), axis=1)

            MAPort.Write(
                nn_parameter_paths[_i],
                ValueFactory.CreateFloatMatrixValue(
                    Array[Array[float]](w.tolist())
                )
            )


            if self.save_network and ((_i == 0) or self.started_saving):
                self.started_saving = True
                checkpoint_path = self.experiment_name
                Path(checkpoint_path).mkdir(parents=True, exist_ok=True)
                if _i == 0:
                    with h5py.File(Path(checkpoint_path + "/weights_checkpoint_" + str(self.checkpoint_nb) + ".hdf5"), mode="w") as f:
                        f.create_dataset(f"w{_i}", data=w)
                else:
                    with h5py.File(Path(checkpoint_path + "/weights_checkpoint_" + str(self.checkpoint_nb) + ".hdf5"), mode="a") as f:
                        f.create_dataset(f"w{_i}", data=w)
                if _i == self.nb_dense_layers-1:
                    self.checkpoint_nb += 1
                    print("SAVED")
                    self.save_network = False
                    self.started_saving = False

    def save_network_weights(self):
        for _i in range(self.nb_dense_layers):
            self.started_saving = True

            w = np.transpose(np.append(self.model_weights[_i * 2], [self.model_weights[_i * 2 + 1]], axis=0))
            w_shape = np.shape(w)
            if _i == self.nb_dense_layers - 1:
                w = np.append(w, np.zeros([self.nb_neurons_per_layer - w_shape[0], w_shape[1]]), axis=0)
            else:
                w = np.append(w, np.zeros([w_shape[0], self.nb_neurons_per_layer + 1 - w_shape[1]]), axis=1)

            checkpoint_path = self.experiment_name
            Path(checkpoint_path).mkdir(parents=True, exist_ok=True)
            if _i == 0:
                with h5py.File(Path(checkpoint_path + "/weights_checkpoint_" + str(self.checkpoint_nb) + ".hdf5"),
                               mode="w") as f:
                    f.create_dataset(f"w{_i}", data=w)
            else:
                with h5py.File(Path(checkpoint_path + "/weights_checkpoint_" + str(self.checkpoint_nb) + ".hdf5"),
                               mode="a") as f:
                    f.create_dataset(f"w{_i}", data=w)
            if _i == self.nb_dense_layers - 1:
                self.checkpoint_nb += 1
                print("SAVED")
                self.save_network = False
                self.started_saving = False

    def input_parser(self):
        while True:
            if self.save_network:
                msg = ''
            else:
                msg = input("Type s to save network weights \n")

            if msg == "s":
                print("SAVING")
                self.save_network = True
            else:
                time.sleep(0.2)
