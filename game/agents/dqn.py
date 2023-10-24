import random
import numpy as np
# import tensorflow as tf
from collections import deque
# from tensorflow import keras
from keras import layers

import codecs
from ..game import Agent
from ..tic_tac_toe import TicTacToeGame, TicTacToeAction, GamePlayer, BOARD_SIZE, BOARD_DIM
# from keras.models import load_model
from tensorflow import keras

from ..tools import getCloestElement, award2
from ..utils import agent_signs


# from tensorflow.keras import layers

def lerp(v, d):
    return v[0] * (1 - d) + v[1] * d

nInputs = 3 * BOARD_SIZE

# Podstawowy Agent DQN z nagrodą cząstkową sieć wejściowa :300
class DQNAgent(Agent):
    def __init__(self, i_agent: int,
                 is_learning: bool = True,
                 learning_rate=1e-3,
                 gamma: float = 0.95,
                 epsilon: float = 0.5,
                 epsilon_end: float = 0.001,
                 epsilon_decay_linear: float = 1 / 2000,
                 pre_training_games: int = 50,
                 experience_replay_batch_size=128,
                 memory_size=10000,
                 reward_draw: float = 5.,
                 reward_win: float = 10.,
                 reward_loss: float = -10.,
                 double_dqn=True, double_dqn_n_games=1,
                 dueling_dqn=True,
                 randomizer = [True, False],
                 seed=42):
        super().__init__(i_agent)
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_end = epsilon_end
        self.epsilon_decay_linear = epsilon_decay_linear
        self.learning_rate = learning_rate
        self.experience_replay_batch_size = experience_replay_batch_size
        self.memory_size = memory_size
        self.double_dqn = double_dqn
        self.double_dqn_n_games = double_dqn_n_games
        self.dueling_dqn = dueling_dqn
        self.randomizer = randomizer
        self.pre_training_games = pre_training_games
        self.seed = seed
        self.reward_draw = reward_draw
        self.reward_win = reward_win
        self.reward_loss = reward_loss
        self.is_learning = is_learning

        self.num_games = -1
        self.n_inputs = nInputs
        self.n_actions = BOARD_SIZE
        self.memory = deque(maxlen=memory_size)
        self.maxLen = BOARD_DIM * BOARD_DIM
        self.maxLen2 = self.maxLen - BOARD_DIM

        self.reward1 = 0

        random.seed(seed)
        self.pre_action = -1
        self.model = self.get_model()
        self.award = 0
        if self.double_dqn:
            self.target_model = self.get_model()
            self.update_target_model()  # Sync weights

    def loadModel(self, path):
        self.model = keras.models.load_model(path)
        # self.model.saving.load_model(path)

    def saveModel(self, path):
        self.model.save(path)

    def new_game(self, game):
        self.reward1 = 0
        self.num_games += 1
        self.stage = None
        self.game_log = []

    def end_game(self, game):
        if not self.is_learning:
            return

        self.commit_log(game, True)
        self.reward1 = 0
        if self.num_games >= self.pre_training_games:
            self.train(self.game_log)

        if (self.experience_replay_batch_size > 0 and
                self.num_games >= self.pre_training_games and
                len(self.memory) >= self.experience_replay_batch_size):
            samples = random.sample(self.memory, self.experience_replay_batch_size)
            self.train(samples)

        self.memory.extend(self.game_log)

        if self.num_games > 0 and self.num_games % self.double_dqn_n_games == 0:
            self.update_target_model()

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
            return award2(game.board.tolist(), i_action,game.board[i_action] ) #+ 1e-20

    # def mini(self,N, mi):
    #     if N >= mi + BOARD_DIM:
    #         return False
    #     return N
    #
    # def maxi(self,N, mi):
    #     if N < mi:
    #         return False
    #     return N
    #
    # def getRow(self,param, param1, param2, param3, tab, agent):
    #     try:
    #         if tab[param] == tab[param1] == tab[param2] == tab[param3] == agent:
    #             return 1
    #     except:
    #         return 0
    #     try:
    #         if tab[param] == tab[param1] == tab[param2] == agent:
    #             return 0.0001
    #     except:
    #         return 0
    #     try:
    #         if(tab[param] == tab[param1] == agent):
    #             return 1e-08
    #     except:
    #         return 0
    #     try:
    #         if(tab[param] == agent):
    #             return 1e-12
    #     except:
    #         return 0
    #
    #     return 0
    #
    #     # try:
    #     # except:
    # #
    # # def getRowOLD(self,param, param1, param2, param3, tab, agent):
    # #     award = 0
    # #
    # #     try:
    # #         # if param and tab[param] == agent:
    # #         award += 10 * (tab[param])
    # #     except:
    # #         pass
    # #     try:
    # #         # if param1 and tab[param] == agent:
    # #         award += 7 * (tab[param1])
    # #     except:
    # #         pass
    # #     try:
    # #         # if param2 and tab[param] == agent:
    # #         award += 3 * (tab[param2])
    # #     except:
    # #         pass
    # #     try:
    # #         # if param3 and tab[param] == agent:
    # #         award +=7 * (tab[param3])
    # #     except:
    # #         pass
    # #     return abs(award)
    #
    # def award2(self,tab, cell, agent):
    #     mi = cell // BOARD_DIM * BOARD_DIM
    #     award = []
    #
    #     award.append(self.getRow(self.mini(cell + 1, mi), self.mini(cell + 2, mi), self.mini(cell + 3, mi),self.mini(cell + 4, mi), tab, agent))
    #     award.append(self.getRow(self.maxi(cell - 1, mi), self.maxi(cell - 2, mi), self.maxi(cell - 3, mi),self.maxi(cell - 4, mi), tab, agent))
    #     award.append(self.getRow(self.mini(cell + BOARD_DIM, self.maxLen2), self.mini(cell + 2 * BOARD_DIM, self.maxLen2),self.mini(cell + 3 * BOARD_DIM, self.maxLen2), self.mini(cell + 4 * BOARD_DIM, self.maxLen2),tab, agent))
    #     award.append(self.getRow(self.maxi(cell - BOARD_DIM, 0), self.maxi(cell - 2 * BOARD_DIM, 0),
    #                     self.maxi(cell - 3 * BOARD_DIM, 0),
    #                     self.maxi(cell - 4 * BOARD_DIM, 0), tab, agent))
    #
    #     award.append(self.getRow(self.mini(cell + BOARD_DIM + 1, self.maxLen2), self.mini(cell + 2 * BOARD_DIM + 2, self.maxLen2),
    #                     self.mini(cell + 3 * BOARD_DIM + 3, self.maxLen2),
    #                     self.mini(cell + 4 * BOARD_DIM + 4, self.maxLen2), tab, agent))
    #
    #     award.append(self.getRow(self.maxi(cell + BOARD_DIM - 1, mi + BOARD_DIM),
    #                     self.maxi(cell + 2 * BOARD_DIM - 2, mi + 2 * BOARD_DIM),
    #                     self.maxi(cell + 3 * BOARD_DIM - 3, mi + 3 * BOARD_DIM),
    #                     self.maxi(cell + 4 * BOARD_DIM - 4, mi + 4 * BOARD_DIM),
    #                     tab, agent))
    #
    #     award.append(self.getRow(self.maxi(self.mini(cell - BOARD_DIM + 1, mi - BOARD_DIM), 0),
    #                     self.maxi(self.mini(cell - 2 * BOARD_DIM + 2, mi - BOARD_DIM * 2), 0),
    #                     self.maxi(self.mini(cell - 3 * BOARD_DIM + 3, mi - BOARD_DIM * 3), 0),
    #                     self.maxi(self.mini(cell - 4 * BOARD_DIM + 4, mi - BOARD_DIM * 4), 0), tab, agent))
    #
    #     award.append(self.getRow(self.maxi(cell - BOARD_DIM - 1, 0), self.maxi(cell - 2 * BOARD_DIM - 2, 0),
    #                     self.maxi(cell - 3 * BOARD_DIM - 3, 0),
    #                     self.maxi(cell - 4 * BOARD_DIM - 4, 0), tab, agent))
    #     x = max(award)
    #     return x
    #
    # def getRow(self,param, param1, param2, param3, tab, agent):
    #     try:
    #         if tab[param] == tab[param1] == tab[param2] == tab[param3] == agent:
    #             return 1
    #     except:
    #         return 0
    #     try:
    #         if tab[param] == tab[param1] == tab[param2] == agent:
    #             return 0.0001
    #     except:
    #         return 0
    #     try:
    #         if(tab[param] == tab[param1] == agent):
    #             return 1e-08
    #     except:
    #         return 0
    #     try:
    #         if(tab[param] == agent):
    #             return 1e-12
    #     except:
    #         return 0
    #
    #     return 0
    #
    #
    # def getCloest(self,param, param1, param2, param3, param4, tab, agent):
    #     try:
    #         if tab[param] == 0 and tab[param1] == agent:
    #             return [param, -4]
    #     except:
    #         pass
    #     try:
    #         if tab[param] == tab[param1] == 0 and tab[param2] == agent:
    #             return [param1, -3]
    #     except:
    #         pass
    #     try:
    #         if tab[param] == tab[param1] == tab[param2] == 0 and tab[param3] == agent:
    #             return [param3, -2]
    #     except:
    #         pass
    #     try:
    #         if tab[param] == tab[param1] == tab[param2] == tab[param3] == 0 or tab[param4] == agent:
    #             return [param3, -1]
    #     except:
    #         pass
    #     # print(agent,tab)
    #     # print(tab[param] == tab[param1] == tab[param2] == tab[param3] == agent or tab[param4])
    #
    #     return [0, 0]


    # def getCloestElement(self, cell, tab, agent):
    #     mi = cell // BOARD_DIM * BOARD_DIM
    #     # award = 0
    #     # award = [self.mini(cell + 1, mi),self.maxi(cell - 1, mi),self.mini(cell + BOARD_DIM, self.maxLen2),self.maxi(cell - BOARD_DIM, 0),
    #     #          self.mini(cell + BOARD_DIM + 1,self.maxi(cell + BOARD_DIM - 1,self.maxi(self.mini(cell - BOARD_DIM + 1,
    #     #          self.maxi(cell - BOARD_DIM - 1, 0),
    #     award = np.array([
    #         self.getCloest(self.mini(cell + 1, mi), self.mini(cell + 2, mi), self.mini(cell + 3, mi),
    #                         self.mini(cell + 4, mi), self.mini(cell + 5, mi), tab, agent),
    #         self.getCloest(self.maxi(cell - 1, mi), self.maxi(cell - 2, mi), self.maxi(cell - 3, mi),
    #                         self.maxi(cell - 4, mi), self.maxi(cell - 5, mi), tab, agent),
    #         self.getCloest(self.mini(cell + BOARD_DIM, self.maxLen2),
    #                         self.mini(cell + 2 * BOARD_DIM, self.maxLen2),
    #                         self.mini(cell + 3 * BOARD_DIM, self.maxLen2),
    #                         self.mini(cell + 4 * BOARD_DIM, self.maxLen2),
    #                         self.mini(cell + 5 * BOARD_DIM, self.maxLen2), tab, agent),
    #         self.getCloest(self.maxi(cell - BOARD_DIM, 0), self.maxi(cell - 2 * BOARD_DIM, 0),
    #                         self.maxi(cell - 3 * BOARD_DIM, 0), self.maxi(cell - 4 * BOARD_DIM, 0),
    #                         self.maxi(cell - 5 * BOARD_DIM, 0), tab, agent),
    #         self.getCloest(self.mini(cell + BOARD_DIM + 1, self.maxLen2),
    #                         self.mini(cell + 2 * BOARD_DIM + 2, self.maxLen2),
    #                         self.mini(cell + 3 * BOARD_DIM + 3, self.maxLen2),
    #                         self.mini(cell + 4 * BOARD_DIM + 4, self.maxLen2),
    #                         self.mini(cell + 5 * BOARD_DIM + 5, self.maxLen2), tab, agent),
    #         self.getCloest(self.maxi(cell + BOARD_DIM - 1, mi + BOARD_DIM),
    #                         self.maxi(cell + 2 * BOARD_DIM - 2, mi + 2 * BOARD_DIM),
    #                         self.maxi(cell + 3 * BOARD_DIM - 3, mi + 3 * BOARD_DIM),
    #                         self.maxi(cell + 4 * BOARD_DIM - 4, mi + 4 * BOARD_DIM),
    #                         self.maxi(cell + 5 * BOARD_DIM - 5, mi + 5 * BOARD_DIM), tab, agent),
    #         self.getCloest(self.maxi(self.mini(cell - BOARD_DIM + 1, mi - BOARD_DIM), 0),
    #                         self.maxi(self.mini(cell - 2 * BOARD_DIM + 2, mi - BOARD_DIM * 2), 0),
    #                         self.maxi(self.mini(cell - 3 * BOARD_DIM + 3, mi - BOARD_DIM * 3), 0),
    #                         self.maxi(self.mini(cell - 4 * BOARD_DIM + 4, mi - BOARD_DIM * 4), 0),
    #                         self.maxi(self.mini(cell - 5 * BOARD_DIM + 5, mi - BOARD_DIM * 5), 0), tab, agent),
    #         self.getCloest(self.maxi(cell - BOARD_DIM - 1, 0), self.maxi(cell - 2 * BOARD_DIM - 2, 0),
    #                         self.maxi(cell - 3 * BOARD_DIM - 3, 0), self.maxi(cell - 4 * BOARD_DIM - 4, 0),
    #                         self.maxi(cell - 5 * BOARD_DIM - 5, 0), tab, agent)])
    #
    #     not_expect = np.array([[0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]])
    #     if np.array_equal(not_expect, award):
    #         return -1
    #     xxx= award.flatten()
    #     arr = np.argmin(xxx)
    #     aaaa = award[arr-1][0]
    #     return aaaa

    # def getCloest(self, param, param1, param2, param3, tab, agent):
    #     try:
    #         if tab[param] == tab[param1] == tab[param2] == tab[param3] == agent:
    #             return 1
    #     except:
    #         return 0
    #     try:
    #         if tab[param] == tab[param1] == tab[param2] == tab[param3] == agent:
    #             return 0.0001
    #     except:
    #         return 0
    #     try:
    #         if tab[param] == tab[param1] == tab[param2] == tab[param3] == agent:
    #             return 1e-08
    #     except:
    #         return 0
    #     try:
    #         if tab[param] == tab[param1] == tab[param2] == tab[param3] == agent:
    #             return 1e-12
    #     except:
    #         return 0
    #
    #     return 0
    # def getCloestElement(self, cell, tab,agent):
    #     mi = cell // BOARD_DIM * BOARD_DIM
    #     award = 0
    #     award = [self.mini(cell + 1, mi),self.maxi(cell - 1, mi),self.mini(cell + BOARD_DIM, self.maxLen2),self.maxi(cell - BOARD_DIM, 0),
    #              self.mini(cell + BOARD_DIM + 1,self.maxi(cell + BOARD_DIM - 1,self.maxi(self.mini(cell - BOARD_DIM + 1,
    #              self.maxi(cell - BOARD_DIM - 1, 0),
    #
    #
    #
    #     award=[
    #     self.getCloest(self.mini(cell + 1, mi), self.mini(cell + 2, mi), self.mini(cell + 3, mi),self.mini(cell + 4, mi), tab, agent),
    #     self.getCloest(self.maxi(cell - 1, mi), self.maxi(cell - 2, mi), self.maxi(cell - 3, mi),self.maxi(cell - 4, mi), tab, agent),
    #     self.getCloest(self.mini(cell + BOARD_DIM, self.maxLen2), self.mini(cell + 2 * BOARD_DIM, self.maxLen2),self.mini(cell + 3 * BOARD_DIM, self.maxLen2), self.mini(cell + 4 * BOARD_DIM, self.maxLen2),tab, agent),
    #     self.getCloest(self.maxi(cell - BOARD_DIM, 0), self.maxi(cell - 2 * BOARD_DIM, 0),self.maxi(cell - 3 * BOARD_DIM, 0),self.maxi(cell - 4 * BOARD_DIM, 0), tab, agent),
    #     self.getCloest(self.mini(cell + BOARD_DIM + 1, self.maxLen2), self.mini(cell + 2 * BOARD_DIM + 2, self.maxLen2),self.mini(cell + 3 * BOARD_DIM + 3, self.maxLen2),self.mini(cell + 4 * BOARD_DIM + 4, self.maxLen2), tab, agent),
    #     self.getCloest(self.maxi(cell + BOARD_DIM - 1, mi + BOARD_DIM),self.maxi(cell + 2 * BOARD_DIM - 2, mi + 2 * BOARD_DIM),self.maxi(cell + 3 * BOARD_DIM - 3, mi + 3 * BOARD_DIM),self.maxi(cell + 4 * BOARD_DIM - 4, mi + 4 * BOARD_DIM),tab, agent),
    #     self.getCloest(self.maxi(self.mini(cell - BOARD_DIM + 1, mi - BOARD_DIM), 0), self.maxi(self.mini(cell - 2 * BOARD_DIM + 2, mi - BOARD_DIM * 2), 0),self.maxi(self.mini(cell - 3 * BOARD_DIM + 3, mi - BOARD_DIM * 3), 0),self.maxi(self.mini(cell - 4 * BOARD_DIM + 4, mi - BOARD_DIM * 4), 0), tab, agent),
    #     self.getCloest(self.maxi(cell - BOARD_DIM - 1, 0), self.maxi(cell - 2 * BOARD_DIM - 2, 0), self.maxi(cell - 3 * BOARD_DIM - 3, 0), self.maxi(cell - 4 * BOARD_DIM - 4, 0), tab, agent)]
    #
    #     return award / 10
    # def get_small_reward(self, tab, cell, agent):
    #     mi = cell // BOARD_DIM * BOARD_DIM
    #     # arr = (tab[max(max(cell - 2 - 2 * BOARD_DIM, mi - 2 * BOARD_DIM), 0): max(min(cell + 3 - 2 * BOARD_DIM, mi - BOARD_DIM), 0)])  + (tab[max(max(cell - 2 - BOARD_DIM, mi - BOARD_DIM), 0): max(min(cell + 3 - BOARD_DIM, mi), 0)]) + (tab[max(cell - 2, mi): min(cell + 3, mi + BOARD_DIM)]) + (tab[max(cell - 2 + BOARD_DIM, mi + BOARD_DIM): min(cell + 3 + BOARD_DIM, mi + 2 * BOARD_DIM)]) + (tab[max(cell - 2 + 2 * BOARD_DIM, mi + 2 * BOARD_DIM): min(cell + 3 + 2 * BOARD_DIM, mi + 4 * BOARD_DIM)])
    #     arr = ((tab[max(max(cell - 2 - 2 * BOARD_DIM, mi - 2 * BOARD_DIM), 0): max(
    #         min(cell + 3 - 2 * BOARD_DIM, mi - BOARD_DIM), 0)]) \
    #           + (tab[max(max(cell - 2 - BOARD_DIM, mi - BOARD_DIM), 0): max(min(cell + 3 - BOARD_DIM, mi), 0)]) \
    #           + (tab[max(cell - 2, mi): min(cell + 3, mi + BOARD_DIM)]) \
    #           + (tab[max(cell - 2 + BOARD_DIM, mi + BOARD_DIM): min(cell + 3 + BOARD_DIM, mi + 2 * BOARD_DIM)]) \
    #           + (tab[max(cell - 2 + 2 * BOARD_DIM, mi + 2 * BOARD_DIM): min(cell + 3 + 2 * BOARD_DIM, mi + 4 * BOARD_DIM)]))
    #     return arr.count(agent) / 10

    def get_model(self):
        input_layer = layers.Input((self.n_inputs,))
        layer = input_layer
        layer = layers.Dense(self.n_inputs * self.n_actions, activation='relu',kernel_initializer=keras.initializers.HeUniform(seed=self.seed))(layer)

        if self.dueling_dqn:
            state_value = layers.Dense(1,kernel_initializer=keras.initializers.HeUniform(seed=self.seed))(layer)
            state_value = layers.Lambda(lambda s: keras.backend.expand_dims(s[:, 0], -1), output_shape=(self.n_actions,))(state_value)
            action_advantage = layers.Dense(self.n_actions,kernel_initializer=keras.initializers.HeUniform(seed=self.seed))(layer)
            action_advantage = layers.Lambda(lambda a: a[:, :] - keras.backend.mean(a[:, :], keepdims=True),output_shape=(self.n_actions,))(action_advantage)
            layer = layers.Add()([state_value, action_advantage])
        else:
            layer = layers.Dense(self.n_actions,kernel_initializer=keras.initializers.HeUniform(seed=self.seed))(layer)

        model = keras.Model(inputs=input_layer, outputs=layer)
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate), loss='mean_squared_error')
        return model

    # def get_model(self):
    #     input_layer = layers.Input((15, 15, 3))
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

    def update_target_model(self):
        if self.double_dqn:
            self.target_model.set_weights(self.model.get_weights())

    def get_model_inputs(self, game: TicTacToeGame):

        inputs = keras.utils.to_categorical(game.board, num_classes=3).reshape((self.n_inputs,))
        assert inputs.shape == (self.n_inputs,)
        return inputs

    def get_action_index(self, action: TicTacToeAction):
        return action.position

    def get_action(self, i_action: int):
        return TicTacToeAction(self.i_agent, i_action)

    def get_legal_actions(self, game_state):
        return game_state[0::3]

    def prepare_log(self, game: TicTacToeGame, action: TicTacToeAction):
        if self.is_learning:
            state = self.get_model_inputs(game).copy()
            i_action = self.get_action_index(action)
            self.stage = (state, i_action)

    def commit_log(self, game: TicTacToeGame, done: bool):
        if self.is_learning and self.stage != None:
            state, i_action = self.stage
            next_state = self.get_model_inputs(game).copy()
            self.reward1 = self.reward1 + self.get_reward(game, i_action) #+ self.award
            print(self.reward1)
            self.game_log.append((state, i_action, self.reward1, next_state, done))
            self.stage = None

    def get_batch_dimension(self, batch_size):
        return (batch_size, nInputs)

    def train(self, batch):
        """Implements Bellman equation."""
        batch_size = len(batch)
        states = np.zeros(self.get_batch_dimension(batch_size))
        next_states = np.zeros(self.get_batch_dimension(batch_size))

        actions = np.zeros((batch_size), dtype=np.int32)
        rewards = np.zeros(batch_size)
        done = np.zeros(batch_size)

        for i, (state, i_action, reward, next_state, d) in enumerate(batch):
            states[i] = state
            next_states[i] = next_state
            actions[i] = i_action
            rewards[i] = reward
            done[i] = 0. if d else 1.

        assert states.shape == (self.get_batch_dimension(batch_size))
        assert next_states.shape == (self.get_batch_dimension(batch_size))

        q_values = self.model.predict(states)
        q_next = self.model.predict(next_states)

        # Remove illegal next actions
        illegal_value = np.min(q_next) - 1
        legal_actions = np.zeros((batch_size, self.n_actions))
        for i, next_state in enumerate(next_states):
            legal_actions[i] = self.get_legal_actions(next_state)

        q_next = legal_actions * q_next - (legal_actions - 1) * illegal_value

        q_targets = q_values.copy()
        batch_index = np.arange(batch_size, dtype=np.int32)
        if self.double_dqn:
            # Current q network selects the action.
            q_next_actions = np.argmax(q_next, axis=1)
            # Use the target network to evaluate the action.
            q_next_target_model = self.target_model.predict(next_states)
            q_targets[batch_index, actions] = rewards + self.gamma * q_next_target_model[
                batch_index, q_next_actions] * done
        else:
            q_targets[batch_index, actions] = rewards + self.gamma * np.max(q_next, axis=1) * done

        assert q_targets.shape == (batch_size, self.n_actions)

        self.model.fit(states, q_targets, verbose=0)

    def getAction(self,game: TicTacToeGame):
        if self.pre_action >= 0 and random.choice(self.randomizer):
            # if self.pre_action < 0:
            moves = getCloestElement(self.pre_action, game.board, agent_signs[self.i_agent])
            if len(moves) > 0:
                # x = random.choice(moves)
                return TicTacToeAction(self.i_agent, moves[0]), moves[1]
            else:
                return random.choice(game.get_legal_actions(self.i_agent)),0
        else:
            return random.choice(game.get_legal_actions(self.i_agent)),0

    def next(self, game: TicTacToeGame) -> bool:
        # Store previous action in action log.
        # that is why we commit here and in end_game().
        self.commit_log(game, False)

        if self.is_learning and (
                self.num_games < self.pre_training_games or
                random.uniform(0, 1) < lerp([self.epsilon, self.epsilon_end], max(0,
                                                                        self.num_games - self.pre_training_games) * self.epsilon_decay_linear)
        ):

            # legal_actions = game.get_legal_actions(self.i_agent)




            action, self.award = self.getAction(game)

            # else:
            #     move = self.getCloestElement(self.pre_action, game.board, self.i_agent)
            #     if move >= 0:
            #         action = TicTacToeAction(self.i_agent, move)
            #     else:
            #         action = random.choice(game.get_legal_actions(self.i_agent))
            #

            self.pre_action = action.position

        else:
            game_state = self.get_model_inputs(game)
            # Predict action based on current game state.
            q_values = self.model.predict(np.array([game_state]))[0]

            assert q_values.shape == (self.n_actions,)

            # Filter invalid actions
            illegal_value = np.min(q_values) - 1
            legal_actions = self.get_legal_actions(game_state)
            action = self.get_action(np.argmax(legal_actions * q_values - (legal_actions - 1) * illegal_value))

        self.prepare_log(game, action)
        return game.next(action)
