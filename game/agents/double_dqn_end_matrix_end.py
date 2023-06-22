from .double_dqn_matrix10_max_reward import DoubleDQNAgentMatrixMaxReward
from ..tic_tac_toe import TicTacToeGame, BOARD_DIM


# Podstawowy Agent DDQN z nagrodą na końću i siecią 10:10:3

class DoubleDQNAgentEndMatrixEnd(DoubleDQNAgentMatrixMaxReward):

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
