from game.agents.human import Human
# from game.agents.dqn_at_last import DQNAgentAtLast
from game.agents.dqn_at_end import DQNAgentAtEnd
from game.agents.dqn_max_reward import DQNAgentMaxReward
# from game.agents.human import Human
from game.tic_tac_toe import TicTacToeGame
from game.utils import play_games, plot_game_results
# from game.agents import RandomAgent, DQNAgent

# agent nagradzany tylko na końcu

dqn_first = DQNAgentAtEnd(i_agent=1,
                         is_learning=True,
                         learning_rate=0.001,
                         gamma=0.8,
                         epsilon=0.3,
                         epsilon_end=0.0001,
                         epsilon_decay_linear=1 / 3000,
                         experience_replay_batch_size=64,
                         pre_training_games=500,
                         memory_size=10000,
                         reward_draw=10.,
                         reward_win=20.,
                         reward_loss=-20.,
                         randomizer=[True],
                         double_dqn=True,
                         double_dqn_n_games=1,
                         dueling_dqn=True,
                         seed=10)

dqn_second = DQNAgentAtEnd(i_agent=0,
                         is_learning=True,
                         learning_rate=0.0001,
                         gamma=0.8,
                         epsilon=0.2,
                         epsilon_end=0.0001,
                         epsilon_decay_linear=1 / 3000,
                         experience_replay_batch_size=64,
                         pre_training_games=500,
                         memory_size=20000,
                         reward_draw=10.,
                         reward_win=20.,
                         reward_loss=-20.,
                         randomizer=[True],
                         double_dqn=True,
                         double_dqn_n_games=1,
                         dueling_dqn=True,
                         seed=11)

# dqn_second = DQNAgentAtEnd(i_agent=1,
#                             is_learning=True,
#                             learning_rate=0.001,
#                             gamma=0.9,
#                             epsilon=0.3,
#                             epsilon_end=0.0005,
#                             epsilon_decay_linear=1 / 3000,
#                             experience_replay_batch_size=64,
#                             pre_training_games=500,
#                             memory_size=10000,
#                             reward_draw=10.,
#                             reward_win=20.,
#                             reward_loss=-20.,
#                             double_dqn=True,
#                             randomizer=[True,True,False],
#                             double_dqn_n_games=1,
#                             dueling_dqn=True,
#                             seed=9)

# human = Human(0)
# randomAgent = RandomAgent(1)
folder="D:\Xagents_DQN"
model1 = '\Agent_first_04.05_At_End.h5'
model2 = '\Agent_second_04.05_At_End.h5'
path_model1 = folder + model1
path_model2 = folder + model2

dqn_first.loadModel(path_model1)
dqn_second.loadModel(path_model2)

dqn_first.model.summary()
dqn_second.model.summary()

for I in range(150):
    results = play_games(lambda: TicTacToeGame(), [dqn_second,dqn_first],1500,paths= [model2,model1], plot=True, debug=False)
    print("kolejne epoki: " + str(I) + "   ---->")
    plot_game_results(results, 2, 100, [model2, model1], " _ " + str(I))

    # model1 = '\Agent_first_04.05_At_End.h5'
    # model2 = '\Agent_second_04.05_At_End.h5'
    # path_model1 = folder + model1
    # path_model2 = folder + model2

    dqn_first.model.summary()
    dqn_second.model.summary()
    dqn_first.saveModel(path_model1)
    dqn_second.saveModel(path_model2)

# Agent_second_MaxReward.h5 <- słaby
# Agent_first__At_End.h5 <- dobry