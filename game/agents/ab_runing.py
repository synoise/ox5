# class ABPruning(Agent):

import random

from .ABP.abpruningai import ABPruningAI
from .ABP.state import State
from ..game import Agent
from ..tic_tac_toe import TicTacToeGame, TicTacToeAction, GamePlayer





class ABPruning(Agent):

    def __init__(self, i_agent: int):
        super().__init__(i_agent)
        self.i_agent = i_agent
        # self.board = None
        current_match = State()
        self.ai = ABPruningAI(current_match)
        self.ai.state.board = current_match.board

    def getLastMove(self,list1,list2):
        for i in range(len(list1)):
            if list1[i] != list2[i//10][i%10]:
                return (i//10, i%10)
        return None



    def next(self, game: TicTacToeGame) -> bool:
        # action = random.choice(game.get_legal_actions(self.i_agent))
        if len(self.ai.state.moves)>0 :
            if self.i_agent == 0:
                self.ai.state.update_move(1, self.getLastMove(game.board,self.ai.state.board))
            else:
                self.ai.state.update_move(0, self.getLastMove(game.board,self.ai.state.board))
        # ai_move1 = [TicTacToeAction(self.i_agent, i) for i in range(len(self.board)) if self.board[i] == GamePlayer.EMPTY]
        # game.board = game.board.reshape((10, 10)).tolist()
        # ai_move = ai.next_move()
        self.ai.state.board = game.board.reshape((10, 10)).tolist() #current_match.board
        ai_move = self.ai.next_move()
        self.ai.state.update_move(self.i_agent, ai_move)
        XX=TicTacToeAction(self.i_agent, ai_move[0]*10+ai_move[1])
        return game.next(XX)
