import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque

from game.agents import DQNAgent
from game.tic_tac_toe import BOARD_SIZE, BOARD_DIM, TicTacToeGame, GamePlayer, TicTacToeAction

# from game.game import Agent
# from tic_tac_toe import Agent
# from tic_tac_toe import TicTacToeGame, TicTacToeAction, GamePlayer, BOARD_SIZE, BOARD_DIM


class DQNAgentPytorch(DQNAgent):
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
        self.pre_training_games = pre_training_games
        self.seed = seed
        self.reward_draw = reward_draw
        self.reward_win = reward_win
        self.reward_loss = reward_loss
        self.is_learning = is_learning

        self.num_games = -1
        self.n_inputs = 3 * BOARD_SIZE
        self.n_actions = BOARD_SIZE
        self.memory = deque(maxlen=memory_size)
        self.sizeL = 4
        self.sizeR = 5
        self.maxLen = BOARD_DIM * BOARD_DIM
        self.maxLen2 = self.maxLen - BOARD_DIM

        # Initialize Q-network
        self.q_network = self.get_model()
        self.target_network = self.get_model()

        # Define loss function and optimizer
        self.loss_function = nn.MSELoss()
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)

        if self.double_dqn:
            self.update_target_network()  # Sync weights

    def get_model(self):
        model = nn.Sequential(
            nn.Linear(self.n_inputs, self.sizeL),
            nn.ReLU(),
            nn.Linear(self.sizeL, self.sizeR),
            nn.ReLU(),
            nn.Linear(self.sizeR, self.n_actions)
        )
        return model

    def update_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())

    def loadModel(self, path):
        self.q_network.load_state_dict(torch.load(path))

    def saveModel(self, path):
        torch.save(self.q_network.state_dict(), path)

    def new_game(self, game):
        self.num_games += 1
        self.stage = None
        self.game_log = []

    def end_game(self, game):
        if not self.is_learning:
            return

        self.commit_log(game, True)

        if self.num_games >= self.pre_training_games:
            self.train(self.game_log)

        # if (self.experience_replay_batch_size > 0 and self.num_games >= self.pre_training_games and
        # if self.epsilon > self.epsilon_end:
        #         self.epsilon -= self.epsilon_decay_linear

        if self.epsilon > self.epsilon_end:
            self.epsilon -= self.epsilon_decay_linear

    def select_action(self, game: TicTacToeGame, player: GamePlayer) -> TicTacToeAction:
        if not self.is_learning:
            return self.select_best_action(game, player)

        state = self.get_state(game, player)
        if np.random.rand() <= self.epsilon:
            return self.select_random_action(game, player)
        else:
            q_values = self.q_network(torch.Tensor(state))
            action_index = torch.argmax(q_values).item()
            action = TicTacToeAction.from_index(action_index)
            return action

    def select_best_action(self, game: TicTacToeGame, player: GamePlayer) -> TicTacToeAction:
        state = self.get_state(game, player)
        q_values = self.q_network(torch.Tensor(state))
        action_index = torch.argmax(q_values).item()
        action = TicTacToeAction.from_index(action_index)
        return action

    def select_random_action(self, game: TicTacToeGame, player: GamePlayer) -> TicTacToeAction:
        available_actions = game.get_available_actions()
        return random.choice(available_actions)

    def get_state(self, game: TicTacToeGame, player: GamePlayer):
        state = np.array([0] * self.n_inputs)
        for i in range(BOARD_DIM):
            for j in range(BOARD_DIM):
                value = game.board[i][j]
                if value == player:
                    state[i * BOARD_DIM + j] = 1
                elif value == game.get_opponent(player):
                    state[i * BOARD_DIM + j + self.maxLen2] = 1
                else:
                    state[i * BOARD_DIM + j + self.maxLen] = 1
        return state

    def commit_log(self, game: TicTacToeGame, is_game_over):
        player = self.player
        reward = 0

        if is_game_over:
            if game.check_winner() == player:
                reward = self.reward_win
            elif game.check_winner() == TicTacToeGame.DRAW:
                reward = self.reward_draw
            else:
                reward = self.reward_loss

        for state, action, next_state in reversed(self.game_log):
            q_values = self.q_network(torch.Tensor(state))
            q_values_next = self.q_network(torch.Tensor(next_state))

            if self.double_dqn:
                next_q_values = self.target_network(torch.Tensor(next_state))
                next_action_index = torch.argmax(q_values_next).item()
                next_action_q_value = next_q_values[next_action_index].item()
            else:
                next_action_q_value = torch.max(q_values_next).item()

            q_values[action.index()] = reward + self.gamma * next_action_q_value

            self.memory.append((state, action, q_values))

        self.game_log = []

    def train(self, game_log):
        if len(self.memory) < self.experience_replay_batch_size:
            return

        batch = random.sample(self.memory, self.experience_replay_batch_size)
        states = torch.Tensor([x[0] for x in batch])
        actions = [x[1].index() for x in batch]
        q_values = torch.Tensor([x[2] for x in batch])

        self.optimizer.zero_grad()
        predictions = self.q_network(states)
        loss = self.loss_function(predictions, q_values)
        loss.backward()
        self.optimizer.step()