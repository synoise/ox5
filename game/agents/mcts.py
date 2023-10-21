import random
import numpy as np
import tensorflow as tf
from collections import deque
from tensorflow import keras
from keras import layers

import codecs
from ..game import Agent
from ..tic_tac_toe import TicTacToeGame, TicTacToeAction, GamePlayer, BOARD_SIZE, BOARD_DIM
# from keras.models import load_model
from tensorflow import keras

from ..tools import getCloestElement
from ..utils import agent_signs
#
# from tensorflow.keras import layers




class MCSTAgent(Agent):
    def __init__(self, state_shape = np.zeros( 10,10,3), num_actions = 200, num_simulations=10, c_puct=1.0, temperature=1.0):
        self.state_shape = state_shape
        self.num_actions = num_actions
        self.num_simulations = num_simulations
        self.c_puct = c_puct
        self.temperature = temperature

        self.root_node = MCTSNode()

    def loadModel(self, path):
        self.model = keras.models.load_model(path)
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



    def get_action_index(self, action: TicTacToeAction):
        return action.position

    def get_action(self, i_action: int):
        return TicTacToeAction(self.i_agent, i_action)





    def get_action(self, state):
        for i in range(self.num_simulations):
            node = self.root_node
            state_copy = np.copy(state)
            # selection
            while not node.is_leaf():
                action, node = node.select(self.c_puct)
                state_copy, _, _ = self.transition(state_copy, action)
            # expansion
            if not node.is_terminal():
                node.expand(state_copy, self.num_actions)
                action, node = node.select(self.c_puct)
                state_copy, _, _ = self.transition(state_copy, action)
            # simulation
            result = self.simulate(state_copy)
            # backpropagation
            while node is not None:
                node.update(result)
                node = node.parent
        action = self.root_node.get_action(self.temperature)
        self.root_node = self.root_node.get_child(action)
        return action

    def transition(self, state, action):
        # perform an action on the state and return the new state, reward, and done flag
        pass

    def simulate(self, state):
        # simulate a game from the current state and return the result (1 if the agent won, -1 if the opponent won, 0 if it's a draw)
        pass

    def getAction(self,game: TicTacToeGame):
            return random.choice(game.get_legal_actions(self.i_agent)),0

    def next(self, game: TicTacToeGame) -> bool:
        action, self.award = self.getAction(game)
        return game.next(action)


class MCTSNode:
    def __init__(self, parent=None, action=None, value=0, visit_count=0):
        self.parent = parent
        self.action = action
        self.children = {}
        self.value = value
        self.visit_count = visit_count

    def is_leaf(self):
        return len(self.children) == 0

    def is_terminal(self):
        return self.visit_count > 0 and self.value != 0

    def select(self, c_puct):
        total_visit_count = sum(child.visit_count for child in self.children.values())
        best_score = float('-inf')
        best_action = None
        best_child = None
        for action, child in self.children.items():
            score = child.value / child.visit_count + c_puct * np.sqrt(np.log(total_visit_count) / child.visit_count)
            if score > best_score:
                best_score = score
                best_action = action
                best_child = child
        return best_action, best_child

    def expand(self, state, num_actions):
        legal_actions = self.get_legal_actions(state, num_actions)
        for action in legal_actions:
            self.children[action] = MCTSNode(parent=self, action=action)

    def update(self, result):
        self.visit_count += 1
        self.value += result

    def get_legal_actions(self, state, num_actions):
        # return the legal actions for the current state
        pass

    def get_action(self, temperature):
        visit_counts = np.array([child.visit_count for child in self.children.values()])
        if temperature == 0:
            action = list(self.children.keys())[np.argmax(visit_counts)]
        else:
            visit_probabilities = visit_counts ** (1 / temperature) / np.sum(visit_counts ** (1 / temperature))
            action = np.random.choice(list(self.children.keys()), p=visit_probabilities)
        return action




