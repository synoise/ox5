# import random
# from copy import deepcopy
# import numpy as np
# import tensorflow as tf
# from collections import deque
# from tensorflow import keras
# # from keras import layers
#
# # from . import DQNAgent
# from ..game import Agent
# from ..tic_tac_toe import TicTacToeGame, TicTacToeAction, GamePlayer, BOARD_SIZE, BOARD_DIM
# from keras.models import load_model
#
# def lerp(v, d):
#     return v[0] * (1 - d) + v[1] * d
#
# class DQNAgentMaxReward(DQNAgent):
#
#     # def get_legal_actions(self, game_state):
#     #     return game_state[0::3]
#
#     def get_reward(self, game: TicTacToeGame, i_action=-1) -> float:
#         if game.is_game_over():
#             winners = game.get_winners()
#             if len(winners) > 1:
#                 return self.reward_draw
#             elif winners[0] == self.i_agent:
#                 return self.reward_win
#             else:
#                 return self.reward_loss
#         else:
#             return self.award
#
