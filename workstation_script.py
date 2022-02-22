"""This script runs the RemoteDDPGAgent on the workstation."""

from EdgeRL_kerasRL2.remote_dqn_agent import RemoteDQNAgent
import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, LeakyReLU
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from rl.memory import SequentialMemory
import threading

def input_parser(*objects_to_close):
    msg = ''
    while msg != 'c':
        msg = input()
        print(msg)
    for close in objects_to_close:
        close.close()

if __name__ == '__main__':
    #tf.compat.v1.disable_eager_execution() #? needed? y/n?
    cyberdyne = '131.234.172.197'
    office = '131.234.124.79'
    local = '127.0.0.1'
    address = office

    nb_actions = 8
    observation_length = 9
    window_length = 1

    nb_layers = 10
    nb_neurons = 127
    leaky_relu_parameter = 0.3
    learning_rate = 5e-5
    memory_buffer_size = 400000
    gamma = 0.85



    model = Sequential()
    model.add(Flatten(input_shape=(window_length, observation_length)))
    for i in range(nb_layers-1):
        model.add(Dense(nb_neurons, activation='linear'))
        model.add(LeakyReLU(alpha=leaky_relu_parameter))
    model.add(Dense(nb_actions,
                    activation='linear'
                    ))


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

    #weight_saver = WeightSaver('TestingWeights', load=False, save_interval=50000)
    threading.Thread(target=input_parser, args=(agent,)).start() # needed?
    agent.start(verbose=2)
