# from game.agents.human import Human
from game.agents.dqn_at_last import DQNAgentAtLast
from game.agents.human import Human
from game.tic_tac_toe import TicTacToeGame
from game.utils import play_games, plot_game_results
from game.agents import RandomAgent, DQNAgent

# agent nagradzany tylko na końcu

dqn_first = DQNAgent(i_agent=0,
                         is_learning=True,
                         learning_rate=0.001,
                         gamma=0.9,
                         epsilon=0.3,
                         epsilon_end=0.0005,
                         epsilon_decay_linear=1 / 3000,
                         experience_replay_batch_size=64,
                         pre_training_games=500,
                         memory_size=10000,
                         reward_draw=5.,
                         reward_win=10.,
                         reward_loss=-10.,
                         double_dqn=True,
                         double_dqn_n_games=1,
                         dueling_dqn=True,
                         seed=4)

dqn_second = DQNAgent(i_agent=1,
                            is_learning=True,
                            learning_rate=0.001,
                            gamma=0.9,
                            epsilon=0.3,
                            epsilon_end=0.0005,
                            epsilon_decay_linear=1 / 3000,
                            experience_replay_batch_size=64,
                            pre_training_games=500,
                            memory_size=10000,
                            reward_draw=5.,
                            reward_win=10.,
                            reward_loss=-10.,
                            double_dqn=True,
                            double_dqn_n_games=1,
                            dueling_dqn=True,
                            seed=3)

# dqn_second = DQNAgent(      i_agent=1,
#                             is_learning=True,
#                             learning_rate=0.005,
#                             gamma=0.99,
#                             epsilon=0.9,
#                             epsilon_end=0.1,
#                             epsilon_decay_linear=1 / 2000,
#                             experience_replay_batch_size=64,
#                             pre_training_games = 40,
#                             memory_size=20000,
#                             reward_draw=5.,
#                             reward_win=10,
#                             reward_loss=-10,
#                             double_dqn=True,
#                             double_dqn_n_games=1,
#                             dueling_dqn=True,
#                             seed=33)

# human = Human(0)
# randomAgent = RandomAgent(1)
model1='\Agent_dqn-first_0_at_end.h5'
model2='\Agent_dqn-second_0_at_end.h5'

folder="D:\Xagents_DQN"
path_model1 = folder + model1
path_model2 = folder + model2

dqn_first.loadModel(path_model1)
dqn_second.loadModel(path_model2)

dqn_first.model.summary()
dqn_second.model.summary()

for I in range(150):
    results = play_games(lambda: TicTacToeGame(), [dqn_first, dqn_second],1500,paths= [model1,model2], plot=True, debug=False)
    print("kolejne epoki: " + str(I) + "   ---->")
    plot_game_results(results, 2,100,[path_model1,path_model2])
    dqn_first.model.summary()
    dqn_second.model.summary()
    dqn_first.saveModel(path_model1)
    dqn_second.saveModel(path_model2)
