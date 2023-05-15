from .dqn_matrix10_max_reward import DQNAgentMatrixMaxReward
from ..tic_tac_toe import TicTacToeGame, BOARD_DIM


# Podstawowy Agent DQN z nagrodą na końću i siecią 10:10:3

class DQNAgentEndMatrixEnd(DQNAgentMatrixMaxReward):

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
            return 0
