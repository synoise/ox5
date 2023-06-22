from tensorflow import keras
from .dqn_matrix10_max_reward import DQNAgentMatrixMaxReward
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
class DoubleDQNAgentMatrixMaxReward(DQNAgentMatrixMaxReward):

    def get_model(self):
        # Input layer
        input_layer = layers.Input(shape=(BOARD_SIZE, BOARD_SIZE, 3))

        # Convolutional layers
        layer = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(input_layer)
        layer = layers.BatchNormalization()(layer)
        layer = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(layer)
        layer = layers.BatchNormalization()(layer)
        layer = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(layer)
        layer = layers.BatchNormalization()(layer)

        # Flatten layer
        layer = layers.Flatten()(layer)

        # Dueling DQN architecture
        if self.dueling_dqn:
            state_value = layers.Dense(1, kernel_initializer=tf.keras.initializers.HeUniform(seed=self.seed))(layer)
            state_value = layers.Lambda(lambda s: tf.expand_dims(s[:, 0], -1), output_shape=(self.n_actions,))(state_value)
            action_advantage = layers.Dense(self.n_actions,kernel_initializer=tf.keras.initializers.HeUniform(seed=self.seed))(layer)
            action_advantage = layers.Lambda(lambda a: a[:, :] - tf.reduce_mean(a[:, :], keepdims=True),output_shape=(self.n_actions,))(action_advantage)
            layer = layers.Add()([state_value, action_advantage])
        else:
            layer = layers.Dense(self.n_actions, kernel_initializer=tf.keras.initializers.HeUniform(seed=self.seed))(layer)
            # output_layer = layers.Dense(self.n_actions, activation='softmax', kernel_initializer=tf.keras.initializers.HeUniform(seed=self.seed))(layer)


        model = tf.keras.Model(inputs=input_layer, outputs=layer)
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),loss=tf.keras.losses.Huber())

        return model

    # def get_model(self):
    #
    #         input_layer = layers.Input(shape=(BOARD_SIZE, BOARD_SIZE, n_in_channels))
    #         layer = input_layer
    #
    #         # convolutional layers
    #         for i in range(n_conv_layers):
    #             layer = layers.Conv2D(n_filters, filter_size, padding='same',kernel_initializer=tf.keras.initializers.HeUniform(seed=self.seed))(layer)
    #             layer = layers.BatchNormalization()(layer)
    #             layer = layers.Activation('relu')(layer)
    #
    #         # flatten layer
    #         layer = layers.Flatten()(layer)
    #
    #         # fully connected layers
    #         for i in range(n_fc_layers):
    #             layer = layers.Dense(fc_size, kernel_initializer=tf.keras.initializers.HeUniform(seed=seed))(layer)
    #             layer = layers.BatchNormalization()(layer)
    #             layer = layers.Activation('relu')(layer)
    #
    #         # output layer
    #         output_layer = layers.Dense(self.n_actions, activation='softmax', kernel_initializer=tf.keras.initializers.HeUniform(seed=self.seed))(layer)
    #
    #         model = tf.keras.Model(inputs=input_layer, outputs=output_layer)
    #         model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate), loss='categorical_crossentropy')
    #         return model

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





