"""This script runs the RemoteDDPGAgent on the workstation."""

from EdgeRL_kerasRL2.remote_dqn_agent import RemoteDQNAgent
import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, LeakyReLU
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from rl.memory import SequentialMemory
import threading
import numpy as np


def input_parser(*objects_to_close):
    msg = ''
    while msg != 'c':
        msg = input()
        print(msg)
    for close in objects_to_close:
        close.close()

if __name__ == '__main__':
    #tf.compat.v1.disable_eager_execution() #? needed? y/n?
    address = # IP address of this workstation

    nb_actions = 8
    observation_length = 14
    window_length = 1

    nb_layers = 10
    dqn_neurons = 90
    #aqtor_neurons = 90
    leaky_relu_parameter = 0.3
    learning_rate = 1e-3
    memory_buffer_size = 400000
    gamma = 0.85

    ### DQN
    model = Sequential()
    model.add(Flatten(input_shape=(window_length, observation_length)))
    for i in range(nb_layers-1):
        model.add(Dense(dqn_neurons, activation='linear'))
        model.add(LeakyReLU(alpha=leaky_relu_parameter))
    raw_q = Dense(nb_actions, activation='linear')(model(model.input))

    model = tf.keras.Model(model.inputs, raw_q)




    weights = model.get_weights()
    for i, w in enumerate(weights):
        weights[i] = 1 * w
    model.set_weights(weights)

    memory = SequentialMemory(
        limit=memory_buffer_size,
        window_length=window_length,
    )

    agent = RemoteDQNAgent(
        # pipeline parameters
        address=address,
        data_port=1030,
        weights_port=1031,
        observation_length=observation_length,
        model=model,
        step_offset=0,

        # agent parameters
        nb_actions=nb_actions,
        gamma=gamma,
        memory=memory,
        batch_size=32,
        target_model_update=0.2,
        memory_interval=1,
        enable_double_dqn=False,
        nb_steps_warmup=32,
        train_interval=1,
        optimizer=Adam,
        learning_rate=learning_rate
    )

    threading.Thread(target=input_parser, args=(agent,)).start()  # needed?
    agent.start(verbose=2)
