from tensorflow import keras
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
learning_rate = 0.001
seed = 42

#NOT END!
#  Agent DQN z nagrodą cząstkowoą i siecią 10:10:3
class DQNAgentMatrixMaxReward(DQNAgent):

    def get_model_inputs(self, game: TicTacToeGame):
        x = keras.utils.to_categorical(game.board, num_classes=3)
        inputs = x.reshape(( BOARD_DIM, BOARD_DIM, 3))
        assert inputs.shape == (10, 10, 3)
        return inputs

    def get_legal_actions(self, game_state):
        return game_state[:, :, 0].flatten()

    def get_batch_dimension(self, batch_size):
        return (batch_size, 10,10,3)


    def get_model(self):

            input_layer = layers.Input(shape=(BOARD_SIZE, BOARD_SIZE, n_in_channels))
            layer = input_layer

            # convolutional layers
            for i in range(n_conv_layers):
                layer = layers.Conv2D(n_filters, filter_size, padding='same',
                                      kernel_initializer=tf.keras.initializers.HeUniform(seed=self.seed))(layer)
                layer = layers.BatchNormalization()(layer)
                layer = layers.Activation('relu')(layer)

            # flatten layer
            layer = layers.Flatten()(layer)

            # fully connected layers
            for i in range(n_fc_layers):
                layer = layers.Dense(fc_size,
                                     kernel_initializer=tf.keras.initializers.HeUniform(seed=seed))(layer)
                layer = layers.BatchNormalization()(layer)
                layer = layers.Activation('relu')(layer)

            # output layer
            output_layer = layers.Dense(self.n_actions, activation='softmax',
                                        kernel_initializer=tf.keras.initializers.HeUniform(seed=self.seed))(layer)

            model = tf.keras.Model(inputs=input_layer, outputs=output_layer)
            model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
                          loss='categorical_crossentropy')
            return model

    # def get_reward(self, game: TicTacToeGame, i_action=-1) -> float:
    #     if game.is_game_over():
    #         winners = game.get_winners()
    #         if len(winners) > 1:
    #             return self.reward_draw
    #         elif winners[0] == self.i_agent:
    #             return self.reward_win
    #         else:
    #             return self.reward_loss
    #     else:
    #         return 0 #





