from tensorflow import keras
from . import DQNAgent
from ..tic_tac_toe import TicTacToeGame


class DQNAgentEndMatrix(DQNAgent):

    def get_model_inputs(self, game: TicTacToeGame):
        mat = game.board.reshape(10, 10)

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
