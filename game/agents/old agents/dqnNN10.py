#
# import numpy as np
#
# from . import DQNAgent
# # from . import DQNAgent
#
# from ..tic_tac_toe import TicTacToeGame, TicTacToeAction, GamePlayer, BOARD_SIZE, BOARD_DIM
#
# from tensorflow import keras
# from ..tools import getCloestElement, award2
#
#
#
# # Agent DQN z nagrodą cząstkową sieć wejściowa : 10:10:3
#
#
#
# nInputs =(BOARD_DIM, BOARD_DIM, 3)
#
# class DQNAgent10x(DQNAgent):
#
#
#     # def get_model_inputs(self, game: TicTacToeGame):
#     #     x = keras.utils.to_categorical(game.board, num_classes=3)
#     #     inputs = x.reshape((1, BOARD_DIM, BOARD_DIM, 3))  # dodajemy wymiar None = 1
#     #     assert inputs.shape == (self.n_inputs,)
#     #     return inputs
#
#
#
#
#     # def train(self, batch):
#     #     """Implements Bellman equation."""
#     #     batch_size = len(batch)
#     #     states = np.zeros(self.get_batch_dimension(batch_size))
#     #     next_states = np.zeros(self.get_batch_dimension(batch_size))
#     #     actions = np.zeros((batch_size), dtype=np.int32)
#     #     rewards = np.zeros(batch_size)
#     #     done = np.zeros(batch_size)
#     #
#     #     for i, (state, i_action, reward, next_state, d) in enumerate(batch):
#     #         states[i] = state
#     #         next_states[i] = next_state
#     #         actions[i] = i_action
#     #         rewards[i] = reward
#     #         done[i] = 0. if d else 1.
#     #
#     #     assert states.shape == (self.get_batch_dimension(batch_size))
#     #     assert next_states.shape == (self.get_batch_dimension(batch_size))
#     #
#     #     q_values = self.model.predict(states)
#     #     q_next = self.model.predict(next_states)
#     #
#     #     # Remove illegal next actions
#     #     illegal_value = np.min(q_next) - 1
#     #     legal_actions = np.zeros((batch_size, self.n_actions))
#     #     for i, next_state in enumerate(next_states):
#     #         legal_actions[i] = self.get_legal_actions(next_state)
#     #
#     #     q_next = legal_actions * q_next - (legal_actions - 1) * illegal_value
#     #
#     #     q_targets = q_values.copy()
#     #     batch_index = np.arange(batch_size, dtype=np.int32)
#     #     if self.double_dqn:
#     #         # Current q network selects the action.
#     #         q_next_actions = np.argmax(q_next, axis=1)
#     #         # Use the target network to evaluate the action.
#     #         q_next_target_model = self.target_model.predict(next_states)
#     #         q_targets[batch_index, actions] = rewards + self.gamma * q_next_target_model[
#     #             batch_index, q_next_actions] * done
#     #     else:
#     #         q_targets[batch_index, actions] = rewards + self.gamma * np.max(q_next, axis=1) * done
#     #
#     #     assert q_targets.shape == (batch_size, self.n_actions)
#     #
#     #     self.model.fit(states, q_targets, verbose=0)
#
