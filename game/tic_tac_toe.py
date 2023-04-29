from enum import IntEnum
from abc import ABC, abstractmethod
import random
import numpy as np
from typing import List
from copy import deepcopy

from .game import Game, Agent, Action
from .tools import checkLines,BOARD_SIZE


# Tic-Tac-Toe implementation

class GamePlayer(IntEnum):
    EMPTY = 0
    NAUGHT = 1
    CROSS = -1


BOARD_DIM = BOARD_SIZE
BOARD_SIZE = BOARD_DIM ** 2


# Easier to just hardcode the lines we have to check for the winner.
# Board dimensions won't change anyways.
# CHECK_LINES = [
#   [0, 1, 2], # rows
#   [3, 4, 5],
#   [6, 7, 8],
#   [0, 3, 6], # cols
#   [1, 4, 7],
#   [2, 5, 8],
#   [0, 4, 8], # diagonals
#   [2, 4, 6]
# ]

# def checkLines(BOARD_DIM1):
#     tab = []
#     # CHECK_LINES = [
#     for i in range(BOARD_DIM1):
#         for j in range(BOARD_DIM1):
#             if j < BOARD_DIM1 - 4:
#                 tab.append([j + i * BOARD_DIM1, 1 + j + i * BOARD_DIM1, 2 + j + i * BOARD_DIM1, 3 + j + i * BOARD_DIM1, 4 + j + i * BOARD_DIM1])  # rows
#
#             d = i * BOARD_DIM1 + j
#
#             if i < BOARD_DIM1 - 4 and j < BOARD_DIM1 - 4:
#                 tab.append([d, d + BOARD_DIM1 + 1, d + BOARD_DIM1 * 2 + 2, d + BOARD_DIM1 * 3 + 3, d + BOARD_DIM1 * 4 + 4])  # rows
#             d = i * BOARD_DIM1 + j
#
#             if 3 < i < BOARD_DIM1 and j < BOARD_DIM1 - 4:
#                 tab.append([ d, d - BOARD_DIM1 + 1, d - BOARD_DIM1 * 2 + 2, d - BOARD_DIM1 * 3 + 3, d - BOARD_DIM1 * 4 + 4])  # rows
#
#     for i in range(BOARD_DIM1**2 - BOARD_DIM1*4):
#         tab.append([i, i + BOARD_DIM1, i + BOARD_DIM1 * 2, i + BOARD_DIM1 * 3, i + BOARD_DIM1 * 4])  # cols
#
#     # if j < BOARD_DIM1 - 5 and i < BOARD_DIM1 - 5:
#     #     tab.append([i * j, i * j + BOARD_DIM1 + 1, i * j + BOARD_DIM1 * 2 + 2, i * j + BOARD_DIM1 * 3 + 3, i * j + BOARD_DIM1 * 4 + 4])  # diagonals
#     # if j < BOARD_DIM1 - 5 and i + 5 < BOARD_DIM1:
#     #     tab.append([i * j + 4, i * j + BOARD_DIM1 + 3, i * j + BOARD_DIM1 * 2 + 2, i * j + BOARD_DIM1 * 3 + 1, i * j + BOARD_DIM1 * 4])  #
#     return tab


CHECK_LINES = checkLines(BOARD_DIM)

# we = 1
#
# for ir in CHECK_LINES:
#     we = we + 1
#     print(we, ir)
#
# print(CHECK_LINES)


def agent_id_to_char(cell: GamePlayer):
    if (cell == GamePlayer.CROSS):
        return "X"
    if (cell == GamePlayer.NAUGHT):
        return "O"
    return " "


class TicTacToeAction(Action):
    def __init__(self, i_agent: int, position: int):
        super().__init__(i_agent)
        self.position = position

    def is_legal(self, game: 'TicTacToeGame') -> bool:
        assert game.board[self.position] == GamePlayer.EMPTY
        return game.board[self.position] == GamePlayer.EMPTY

    def run(self, game: 'TicTacToeGame'):
        assert self.is_legal(game)
        game.board[self.position] = game.players[self.i_agent]


class TicTacToeGame(Game):
    def __init__(self):
        super().__init__(n_agents=2)
        self.players = [GamePlayer.NAUGHT, GamePlayer.CROSS]
        self.board = np.ndarray(shape=(1, BOARD_SIZE), dtype=int)[0]
        self.board.fill(GamePlayer.EMPTY)

    def is_game_over(self) -> bool:
        for indexes in CHECK_LINES:
            line = [self.board[i] for i in indexes]
            if (line[0] != GamePlayer.EMPTY and line[0] == line[1] == line[2]== line[3] == line[4]):
                return True

        for value in self.board:
            if (value == GamePlayer.EMPTY):
                return False

        return True

    def get_score(self, i_agent: int) -> int:
        winner = self.get_winner()
        if (winner == None):
            return False
        if (winner == self.players[i_agent]):
            return 1
        return 0

    def get_winner(self) -> GamePlayer:
        for indexes in CHECK_LINES:
            line = [self.board[i] for i in indexes]
            if (line[0] != GamePlayer.EMPTY and line[0] == line[1] == line[2]== line[3]== line[4]):
                return line[0]
        return None

    def get_legal_actions(self, i_agent: int) -> List[TicTacToeAction]:
        return [TicTacToeAction(i_agent, i) for i in range(len(self.board)) if self.board[i] == GamePlayer.EMPTY]

    def setBoard(self,x,y) -> int:
        agent_id_to_char(self.board[x * BOARD_DIM + y ])

    def get_hash(self) -> int:
        # print(123)
        res = 0
        for i in range(BOARD_SIZE):
            # res *= 2
            res += self.board[i]
        return res

    def __str__(self) -> str:
        result = "\n"
        for y in range(BOARD_DIM):
            if y > 0:
                result += "\n"  # ""\n-----\n"
            for x in range(BOARD_DIM):
                if x > 0:
                    result += "|"
                result += agent_id_to_char(self.board[x + y * BOARD_DIM])
        return result
