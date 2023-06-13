from tensorflow import keras

from game.agents.dqn_matrix10_max_reward import DQNAgentMatrixMaxReward
from . import DQNAgent
from ..tic_tac_toe import TicTacToeGame, BOARD_DIM
from keras import layers
from ..tools import BOARD_SIZE
import tensorflow as tf

n_in_channels = 3
n_conv_layers = 8
n_filters = 64
filter_size = 5
n_fc_layers = 1
fc_size = 64
n_actions = BOARD_SIZE * BOARD_SIZE
# learning_rate = 0.001
seed = 42

#NOT END!
#  Agent DQN z nagrodą cząstkowoą i siecią 10:10:3
class DQNAtemp(DQNAgentMatrixMaxReward):





    def get_model(self):

            input_layer = layers.Input(shape=(BOARD_SIZE, BOARD_SIZE, n_in_channels))
            layer = input_layer

            # convolutional layers
            for i in range(n_conv_layers):
                layer = layers.Conv2D(n_filters, filter_size, padding='same',kernel_initializer=tf.keras.initializers.HeUniform(seed=self.seed))(layer)
                layer = layers.BatchNormalization()(layer)
                layer = layers.Activation('relu')(layer)

            # flatten layer
            layer = layers.Flatten()(layer)

            # fully connected layers
            for i in range(n_fc_layers):
                layer = layers.Dense(fc_size, kernel_initializer=tf.keras.initializers.HeUniform(seed=seed))(layer)
                layer = layers.BatchNormalization()(layer)
                layer = layers.Activation('relu')(layer)

            # output layer
            output_layer = layers.Dense(self.n_actions, activation='softmax', kernel_initializer=tf.keras.initializers.HeUniform(seed=self.seed))(layer)

            model = tf.keras.Model(inputs=input_layer, outputs=output_layer)
            model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate), loss='categorical_crossentropy')
            return model






