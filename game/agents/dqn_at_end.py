from . import DQNAgent
from ..tic_tac_toe import TicTacToeGame

# Podstawowy Agent DQN z nagrodą na końcu sieć wejściowa - stara:300
class DQNAgentAtEnd(DQNAgent):

    def get_reward(self, game: TicTacToeGame, i_action=-1) -> float:
        if game.is_game_over():
            self.reward1 = 0
            winners = game.get_winners()
            if len(winners) > 1:
                return self.reward_draw
            elif winners[0] == self.i_agent:
                return self.reward_win
            else:
                return self.reward_loss
        else:
            return 0
