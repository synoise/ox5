# from game.agents.human import Human
# from game.agents.dqn_at_last import DQNAgentAtLast
from game.agents import DQNAgent
from game.agents.dqn_at_end import DQNAgentAtEnd
# from game.agents.dqn_max_reward import DQNAgentMaxReward
# from game.agents.human import Human
# from game.agents.human import Human
from game.tic_tac_toe import TicTacToeGame
from game.utils import play_games, plot_game_results
from game.save_stats import SaveStats

# from game.agents import RandomAgent, DQNAgent

stats = SaveStats()

seed1, seed2 = stats.loadStats('Agent_DQN_Stats.json')


# seed1 =38
# seed2 =38

def initiateAgents():
    dqn_first1 = DQNAgent(i_agent=0,
                          is_learning=True,
                          learning_rate=0.001,
                          gamma=0.8,
                          epsilon=0.3,
                          epsilon_end=0.0001,
                          epsilon_decay_linear=1 / 3000,
                          experience_replay_batch_size=64,
                          pre_training_games=500,
                          memory_size=20000,
                          reward_draw=50.,
                          reward_win=100.,
                          reward_loss=-100.,
                          randomizer=[True,True,False],
                          double_dqn=True,
                          double_dqn_n_games=1,
                          dueling_dqn=True,
                          seed=seed1)

    # agent nagradzany tylko na końcu
    dqn_second1 = DQNAgentAtEnd(i_agent=1,
                                is_learning=True,
                                learning_rate=0.001,
                                gamma=0.8,
                                epsilon=0.2,
                                epsilon_end=0.0001,
                                epsilon_decay_linear=1 / 3000,
                                experience_replay_batch_size=64,
                                pre_training_games=500,
                                memory_size=20000,
                                reward_draw=50.,
                                reward_win=100.,
                                reward_loss=-100.,
                                randomizer=[True,False],
                                double_dqn=True,
                                double_dqn_n_games=1,
                                dueling_dqn=True,
                                seed=seed2)
    return dqn_first1, dqn_second1


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
# human0 = Human(0)
# human1 = Human(1)
# randomAgent = RandomAgent(1)
folder = "D:\Xagents_DQN"
# model1 = '\MATRIX_first_11.05_Gross.h5'
# second2 = '\MATRIX_first_11.05_Gross.h5'
# second2 = '\MATRIX_second_11.05_Gross.h5'
first1 = '\Agent_first_15.05_Gross.h5'
second2 = '\Agent_second_15.05_At_End.h5'
# first1 = '\Agent_first_04.05_At_End.h5'
# second2 = '\Agent_first_04.05_At_End.h5'
# first1 = '\Agent_second__At_End.h5'
# second2 = '\Agent_second__At_End.h5'
# second2 = '\Agent_second__At_End.h5'
path_first1 = folder + first1
path_second2 = folder + second2

for I in range(150):
    dqn_first, dqn_second = initiateAgents()
    dqn_first.loadModel(path_first1)
    dqn_second.loadModel(path_second2)
    seed1 += 1
    dqn_first.model.summary()
    dqn_second.model.summary()

    results = play_games(lambda: TicTacToeGame(), [dqn_first, dqn_second], 1500, paths=[first1, second2], plot=True, debug=True)
    print("kolejne epoki: " + str(I) + "  ----> seed1:" + str(seed1) + " -  seed2:" + str(seed2))
    plot_game_results(results, 2, 100, [first1, second2], " _ " + str(I), stats.colors)
    dqn_first.saveModel(path_first1)
    dqn_second.saveModel(path_second2)
    seed2 += 1
    stats.saveStats('Agent_DQN_Stats.json', seed1, first1, seed2, second2, results, 1500)
    del dqn_first
    del dqn_second

# Agent_second_MaxReward.h5 <- słaby
# Agent_first__At_End.h5 <- dobry
