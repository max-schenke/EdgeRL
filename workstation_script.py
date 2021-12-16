"""This script runs the RemoteDDPGAgent on the workstation."""

from remote_dqn_agent import RemoteDQNAgent
import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Input, Concatenate, LeakyReLU
from callbacks import TrainInterrupter, LearningRateFileScheduler, WeightSaver
from tensorflow.keras import Sequential, Model, initializers
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
    # tf.compat.v1.disable_eager_execution() #? needed? y/n?
    cyberdyne = '131.234.172.197'
    office = '131.234.124.79'
    local = '127.0.0.1'
    address = office

    nb_actions = 8
    state_length = 9
    window_length = 1

    nb_layers = 10
    nb_neurons = 500
    leaky_relu_parameter = 0.3
    learning_rate = 2e-5
    memory_buffer_size = 400000
    gamma = 0.85
    measurement_size = 4 + state_length  # = 1t + 9s + 1a + 1r + 1d



    model = Sequential()
    model.add(Flatten(input_shape=(window_length, state_length)))
    for i in range(nb_layers-1):
        model.add(Dense(nb_neurons, activation='linear'))
        model.add(LeakyReLU(alpha=leaky_relu_parameter))
    model.add(Dense(nb_actions,
                    activation='linear'
                    ))
    print(model.get_weights())


    memory = SequentialMemory(
        limit=memory_buffer_size,
        window_length=window_length,
    )

    agent = RemoteDQNAgent(
        # pipeline parameters
        address=address,
        data_port=1001,
        weights_port=1002,
        measurement_size=measurement_size,
        model=model,
        step_offset=0,

        # agent parameters
        nb_actions=nb_actions,
        gamma=gamma,
        memory=memory,
        batch_size=4,
        target_model_update=0.2,
        memory_interval=1,
        enable_double_dqn=False,
        nb_steps_warmup=20,
        train_interval=1,
    )

    #weight_saver = WeightSaver('TestingWeights', load=False, save_interval=50000)
    agent.compile(Adam(lr=learning_rate))
    threading.Thread(target=input_parser, args=(agent,)).start()

    agent.start(verbose=2)
