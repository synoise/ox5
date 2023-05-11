from tensorflow import keras
from . import DQNAgent
from ..tic_tac_toe import TicTacToeGame, BOARD_DIM
from keras import layers

class DQNAgentEndMatrix(DQNAgent):
    # def get_model(self):
    #     input_layer = layers.Input((BOARD_DIM, BOARD_DIM, 3))
    #     layer = input_layer
    #     layer = layers.Conv2D(32, (5, 5), strides=(1, 1), padding='same', activation='relu')(layer)
    #
    #     for i in range(self.n_layers):
    #         layer = layers.Conv2D(32, (3, 3), strides=(1, 1), padding='same', activation='relu')(layer)
    #
    #     layer = layers.Conv2D(1, (1, 1), strides=(1, 1), padding='same', activation='linear')(layer)
    #     layer = layers.Flatten()(layer)
    #     layer = layers.Dense(225, activation='softmax')(layer)
    #
    #     model = keras.Model(inputs=input_layer, outputs=layer)
    #     model.compile(optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate),
    #                   loss='categorical_crossentropy')
    #     return model

    def get_model_inputs(self, game: TicTacToeGame):
        mat = game.board.reshape(BOARD_DIM, BOARD_DIM)

        inputs = keras.utils.to_categorical(mat, num_classes=3).reshape((self.n_inputs,))
        assert inputs.shape == (self.n_inputs,)
        return inputs

    def get_reward(self, game: TicTacToeGame, i_action=-1) -> float:
        if game.is_game_over():
            winners = game.get_winners()
            if len(winners) > 1:
                return self.reward_draw
            elif winners[0] == self.i_agent:
                return self.reward_win
            else:
                return self.reward_loss
        else:
            return 0 #




        import tensorflow as tf
        from tensorflow.keras import layers

        # class GomokuModel:
        #     def __init__(self):
        #         self.board_width = 20
        #         self.board_height = 20
        #         self.n_in_channels = 3
        #         self.n_conv_layers = 8
        #         self.n_filters = 64
        #         self.filter_size = 5
        #         self.n_fc_layers = 1
        #         self.fc_size = 64
        #         self.n_actions = self.board_width * self.board_height
        #         self.learning_rate = 0.001
        #         self.seed = 42
        #
        #     def get_model(self):
        #         input_layer = layers.Input(shape=(self.board_height, self.board_width, self.n_in_channels))
        #         layer = input_layer
        #
        #         # convolutional layers
        #         for i in range(self.n_conv_layers):
        #             layer = layers.Conv2D(self.n_filters, self.filter_size, padding='same',
        #                                   kernel_initializer=tf.keras.initializers.HeUniform(seed=self.seed))(layer)
        #             layer = layers.BatchNormalization()(layer)
        #             layer = layers.Activation('relu')(layer)
        #
        #         # flatten layer
        #         layer = layers.Flatten()(layer)
        #
        #         # fully connected layers
        #         for i in range(self.n_fc_layers):
        #             layer = layers.Dense(self.fc_size,
        #                                  kernel_initializer=tf.keras.initializers.HeUniform(seed=self.seed))(layer)
        #             layer = layers.BatchNormalization()(layer)
        #             layer = layers.Activation('relu')(layer)
        #
        #         # output layer
        #         output_layer = layers.Dense(self.n_actions, activation='softmax',
        #                                     kernel_initializer=tf.keras.initializers.HeUniform(seed=self.seed))(layer)
        #
        #         model = tf.keras.Model(inputs=input_layer, outputs=output_layer)
        #         model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
        #                       loss='categorical_crossentropy')
        #         return model

