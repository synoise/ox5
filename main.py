# from game.agents.human import Human
from game.agents.dqn_offen import DQNAgentOffen
from game.tic_tac_toe import TicTacToeGame, GamePlayer, TicTacToeAction
from game.utils import play_games, plot_game_results
from game.agents import RandomAgent, DQNAgent

dqn_first = DQNAgentOffen(i_agent=0,
                     is_learning=True,
                     learning_rate=1e-3,
                     gamma=0.95,
                     epsilon=0.5,
                     epsilon_end=0.001,
                     epsilon_decay_linear=1 / 2000,
                     experience_replay_batch_size=128,
                     pre_training_games = 40,
                     memory_size=10000,
                     reward_draw=5.,
                     reward_win=10.,
                     reward_loss=-10.,
                     double_dqn=True,
                     double_dqn_n_games=1,
                     dueling_dqn=True,
                     seed=42)

dqn_second = DQNAgentOffen(i_agent=1,
                           is_learning=True,
                           learning_rate=0.005,
                           gamma=0.99,
                           epsilon=0.9,
                           epsilon_end=0.1,
                           epsilon_decay_linear=1 / 2000,
                           experience_replay_batch_size=64,
                           pre_training_games = 40,
                           memory_size=20000,
                           reward_draw=5.,
                           reward_win=10,
                           reward_loss=-10,
                           double_dqn=True,
                           double_dqn_n_games=1,
                           dueling_dqn=True,
                           seed=42)

# human = Human(0)
randomAgent = RandomAgent(1)
path_model1 = 'D:\Xagents_DQN\dqnagent-first1.h5'
path_model2 = 'D:\Xagents_DQN\dqnagent-second2.h5'

dqn_first.loadModel(path_model1)
# dqn_second.loadModel(path_model2)
dqn_first.model.summary()
dqn_second.model.summary()

for I in range(150):
    results = play_games(lambda: TicTacToeGame(), [dqn_first, randomAgent], 1500, plot = True)
    print("epoka: " + str(I) + "   ---->")
    plot_game_results(results, 2)
    dqn_first.saveModel(path_model1)
    dqn_second.saveModel(path_model2)

