from rl.agents import DQNAgent
import numpy as np
import time
from interface import Interface
import threading
from collections import deque
import h5py
import base64
import io
import csv
from rl.callbacks import TrainEpisodeLogger, TrainIntervalLogger, CallbackList
from tensorflow.keras.callbacks import History
import socket
import struct
import select


class RemoteDQNAgent(DQNAgent):
    """Keras-rl DQNAgent that receives recorded experiences via a tcp-interface, trains on the whole batch of data
    and sends the updated actor-weights back.
    """
    def __init__(self,
                 address=None,
                 data_port=1001,
                 weights_port=1002,
                 measurement_size=None,
                 weights_path=None,
                 step_offset=0,
                 **kwargs):
        """
        Args:
            address(str): IP-address of this host
            port(int): Port for the connection
            weights(path): Path for initial weights to load. Also if a name is passed the weights are saved after each
                            training episode with the same name. Don not add .h5 at the end.
            max_recording_length(int): Maximum number of steps of one recording. All experiences that exceed this number
                                       of steps will be ignored
            weights_save_interval: Number of training steps after the weights should be saved with the ending of number
                                    of iterations. This is done after training an episode and the value should be larger
                                    than the max_recording_length.
            step_offset: if the learning is continued on a pre-trained agent this value is the number of pre-trained
                        steps to get the correct number of training steps
            kwargs: Further arguments of the superclass DQNAgent
        """
        super().__init__(**kwargs)

        self.weights_path = weights_path
        self.address = address
        self.data_port = data_port
        self.measurement_size = measurement_size
        # buffer size is in bytes, float has size of 4 bytes:
        self.data_send_buffer_size = int(1024 // (self.measurement_size * 4)) * self.measurement_size * 4
        self.weights_port = weights_port
        self.close_agent = False

        self.model_weights = self.model.get_weights()
        self._step_offset = step_offset

    def start(self, verbose=0, callbacks=None):
        """Opens the socket on the interface and the training starts.

        Args:
            verbose(int): 0: No logging. 1: Interval Logging 2: Episode based logging
            callbacks(list): List of callbacks.
        """

        # create socket and bind, data receiving connection
        print("Starting remote RL server")
        data_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)  # TCP
        weights_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)  # TCP
        data_socket.bind((self.address, self.data_port))
        weights_socket.bind((self.address, self.weights_port))
        print("Waiting for test bench to establish connection")
        data_socket.listen(1)
        weights_socket.listen(1)

        self.data_conn, data_addr = data_socket.accept()
        self.data_conn.setblocking(False)

        self.weights_conn, weights_addr = weights_socket.accept()
        #self.weights_conn.setblocking(False)

        print("XIL API connection established")

        # convert float data list to bytes
        self.weightBufferSize = 1024
        self.architecture = []
        for _layer in self.model_weights:
            self.architecture.append(np.shape(_layer))
        arch_list = list(sum(self.architecture, ()))
        b = bytes()
        b = b.join((struct.pack('f', val) for val in arch_list))
        # send bytes
        if (len(b) < self.weightBufferSize):
            self.weights_conn.send(b)
        else:
            for i in range(0, len(b) // self.weightBufferSize):
                self.weights_conn.send(b[i * self.weightBufferSize:i * self.weightBufferSize + self.weightBufferSize])

            if len(b) > ((i + 1) * self.weightBufferSize):
                self.weights_conn.send(b[(i + 1) * self.weightBufferSize:])
        time.sleep(5)
        print("Done sending architecture")


        for _i, _layer_dim in enumerate(self.architecture):
            if len(_layer_dim) > 0:
                _layer = np.ndarray.flatten(self.model_weights[_i])
            else:
                _layer = self.model_weights[_i]
            b = bytes()
            b = b.join((struct.pack('f', val) for val in _layer))

            # send bytes
            if (len(b) < self.weightBufferSize):
                self.weights_conn.send(b)
            else:
                for i in range(0, len(b) // self.weightBufferSize):
                    self.weights_conn.send(b[i * self.weightBufferSize:i * self.weightBufferSize + self.weightBufferSize])

                if len(b) > ((i + 1) * self.weightBufferSize):
                    self.weights_conn.send(b[(i + 1) * self.weightBufferSize:])
        print("Done sending weights")

        self._train(verbose=verbose)


    def _train(self, verbose=0, log_interval=10000):
        """Main training loop, episodes do not exist, therefore we have no callbacks"""
        self.training = True

        self._on_train_begin()  # callback call
        # Training Loop

        data_ready = select.select([self.data_conn], [], [])
        if data_ready:
            binary_data = self.data_conn.recv(self.data_send_buffer_size)
            float_data = np.frombuffer(binary_data, dtype=np.float32)
            episode_data = np.reshape(float_data, (self.measurement_size, -1), order="F")  # seems to be fine
            for _i in range(len(episode_data[0])):
                self.memory.append(episode_data[1:10, _i], # state
                                   (episode_data[10, _i]).astype(int), # action
                                   episode_data[11, _i], # reward
                                   bool(episode_data[12, _i]),# done
                                   {},
                                   training=self.training)
            self.step += 1
        print("STARTING")
        local_buffer = None
        while not self.close_agent:

            # listen for data
            print("receiving")
            while select.select([self.data_conn], [], [], 0.0)[0]:
                binary_data = self.data_conn.recv(self.data_send_buffer_size)
                float_data = np.frombuffer(binary_data, dtype=np.float32)
                data_len = len(float_data)
                if data_len % self.measurement_size:
                    full_dates_len = (data_len // self.measurement_size) * self.measurement_size
                    if local_buffer is None:
                        local_buffer = float_data[full_dates_len:]
                        float_data = float_data[:full_dates_len]
                    else:
                        try:
                            local_buffer = np.append(local_buffer, float_data[:-full_dates_len])
                            float_data = list(np.append(local_buffer, float_data[-full_dates_len:]))
                            local_buffer = None
                        except:
                            raise
                try:
                    episode_data = np.reshape(float_data, (self.measurement_size, -1), order="F")  # seems to be fine
                except:
                    raise
                for _i in range(len(episode_data[0])):
                    self.memory.append(episode_data[1:10, _i],
                                       (episode_data[10, _i]).astype(int),
                                       episode_data[11, _i],
                                       bool(episode_data[12, _i]),
                                       {},
                                       training=self.training)

                self.step += 1  # steps in memory

            if self.close_agent:
                return

            print("sending")
            # optional: outsource SendingWeights2ControlDesk to other process?
            if select.select([self.weights_conn], [], [], 0.0)[0]:
                self.weights_conn.recv(1024)
                self.model_weights = self.model.get_weights()
                for _i, _layer_dim in enumerate(self.architecture):
                    if len(_layer_dim) > 0:
                        _layer = np.ndarray.flatten(self.model_weights[_i])
                    else:
                        _layer = self.model_weights[_i]
                    b = bytes()
                    b = b.join((struct.pack('f', val) for val in _layer))

                    # send bytes
                    if (len(b) < self.weightBufferSize):
                        self.weights_conn.send(b)
                    else:
                        for i in range(0, len(b) // self.weightBufferSize):
                            self.weights_conn.send(b[i * self.weightBufferSize:(i + 1) * self.weightBufferSize])

                        if len(b) > ((i + 1) * self.weightBufferSize):
                            self.weights_conn.send(b[(i + 1) * self.weightBufferSize:])
            metrics = self.backward()


    def close(self):
        """Closing function to terminate the threads and to close the interface"""
        self.close_agent = True
        if hasattr(self, 'conn'):
            self.weights_conn.close()
            self.data_conn.close()


    def backward(self):
        # Storing of data into the memory has been outsourced from this fcn because one at a time is too few

        metrics = [np.nan for _ in self.metrics_names]
        if not self.training:
            # We're done here. No need to update the experience memory since we only use the working
            # memory to obtain the state over the most recent observations.
            return metrics

        # Train the network on a single stochastic batch.
        if self.step > self.nb_steps_warmup and self.step % self.train_interval == 0:
            experiences = self.memory.sample(self.batch_size)
            assert len(experiences) == self.batch_size

            # Start by extracting the necessary parameters (we use a vectorized implementation).
            state0_batch = []
            reward_batch = []
            action_batch = []
            terminal1_batch = []
            state1_batch = []
            for e in experiences:
                state0_batch.append(e.state0)
                state1_batch.append(e.state1)
                reward_batch.append(e.reward)
                action_batch.append(e.action)
                terminal1_batch.append(0. if e.terminal1 else 1.)

            # Prepare and validate parameters.
            state0_batch = self.process_state_batch(state0_batch)
            state1_batch = self.process_state_batch(state1_batch)
            terminal1_batch = np.array(terminal1_batch)
            reward_batch = np.array(reward_batch)
            assert reward_batch.shape == (self.batch_size,)
            assert terminal1_batch.shape == reward_batch.shape
            assert len(action_batch) == len(reward_batch)

            # Compute Q values for mini-batch update.
            if self.enable_double_dqn:
                # According to the paper "Deep Reinforcement Learning with Double Q-learning"
                # (van Hasselt et al., 2015), in Double DQN, the online network predicts the actions
                # while the target network is used to estimate the Q value.
                q_values = self.model.predict_on_batch(state1_batch)
                assert q_values.shape == (self.batch_size, self.nb_actions)
                actions = np.argmax(q_values, axis=1)
                assert actions.shape == (self.batch_size,)

                # Now, estimate Q values using the target network but select the values with the
                # highest Q value wrt to the online model (as computed above).
                target_q_values = self.target_model.predict_on_batch(state1_batch)
                assert target_q_values.shape == (self.batch_size, self.nb_actions)
                q_batch = target_q_values[range(self.batch_size), actions]
            else:
                # Compute the q_values given state1, and extract the maximum for each sample in the batch.
                # We perform this prediction on the target_model instead of the model for reasons
                # outlined in Mnih (2015). In short: it makes the algorithm more stable.
                target_q_values = self.target_model.predict_on_batch(state1_batch)
                assert target_q_values.shape == (self.batch_size, self.nb_actions)
                q_batch = np.max(target_q_values, axis=1).flatten()
            assert q_batch.shape == (self.batch_size,)

            targets = np.zeros((self.batch_size, self.nb_actions))
            dummy_targets = np.zeros((self.batch_size,))
            masks = np.zeros((self.batch_size, self.nb_actions))

            # Compute r_t + gamma * max_a Q(s_t+1, a) and update the target targets accordingly,
            # but only for the affected output units (as given by action_batch).
            discounted_reward_batch = self.gamma * q_batch
            # Set discounted reward to zero for all states that were terminal.
            discounted_reward_batch *= terminal1_batch
            assert discounted_reward_batch.shape == reward_batch.shape
            Rs = reward_batch + discounted_reward_batch
            for idx, (target, mask, R, action) in enumerate(zip(targets, masks, Rs, action_batch)):
                target[action] = R  # update action with estimated accumulated reward
                dummy_targets[idx] = R
                mask[action] = 1.  # enable loss for this specific action
            targets = np.array(targets).astype('float32')
            masks = np.array(masks).astype('float32')

            # Finally, perform a single update on the entire batch. We use a dummy target since
            # the actual loss is computed in a Lambda layer that needs more complex input. However,
            # it is still useful to know the actual target to compute metrics properly.
            ins = [state0_batch] if type(self.model.input) is not list else state0_batch
            metrics = self.trainable_model.train_on_batch(ins + [targets, masks], [dummy_targets, targets])
            metrics = [metric for idx, metric in enumerate(metrics) if idx not in (1, 2)]  # throw away individual losses
            metrics += self.policy.metrics
            if self.processor is not None:
                metrics += self.processor.metrics

        if self.target_model_update >= 1 and self.step % self.target_model_update == 0:
            self.update_target_model_hard()

        return metrics
