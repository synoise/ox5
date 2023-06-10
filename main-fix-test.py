# from game.agents.human import Human
# from game.agents.dqn_at_last import DQNAgentAtLast
# from game.agents.dqn_at_end import DQNAgentAtEnd
from game.agents.dqn_matrix10_max_reward import DQNAgentMatrixMaxReward
from game.agents.dqn_end_matrix_end import DQNAgentEndMatrixEnd
from game.save_stats import SaveStats
# from game.agents.dqn_max_reward import DQNAgentMaxReward
# from game.agents.human import Human
from game.agents.human import Human
from game.tic_tac_toe import TicTacToeGame
from game.utils import play_games, plot_game_results
# from game.agents import RandomAgent, DQNAgent
import gc

# agent nagradzany tylko na końcu

stats = SaveStats()
seed1, seed2 = stats.loadStats('./stats/fixed_DDQN_10N10x3_Stats.json')

def initiateAgents():
    dqn_first1 = DQNAgentMatrixMaxReward(i_agent=0,
                                         is_learning=True,
                                         learning_rate=0.0001,
                                         gamma=0.99,
                                         epsilon=0.6,
                                         epsilon_end=0.0001,
                                         epsilon_decay_linear=1 / 3000,
                                         experience_replay_batch_size=64,
                                         pre_training_games=50,
                                         memory_size=50000,
                                         reward_draw=500.,
                                         reward_win=100.,
                                         reward_loss=-100.,
                                         randomizer=[True],
                                         double_dqn=True,
                                         double_dqn_n_games=1,
                                         dueling_dqn=True,
                                         seed=seed1)

    dqn_second1 = DQNAgentEndMatrixEnd(i_agent=1,
                                      is_learning=True,
                                      learning_rate=0.001,
                                      gamma=0.95,
                                      epsilon=0.6,
                                      epsilon_end=0.0001,
                                      epsilon_decay_linear=0.0001,
                                      experience_replay_batch_size=64,
                                      pre_training_games=50,
                                      memory_size=50000,
                                      reward_draw=500.,
                                      reward_win=100.,
                                      reward_loss=-100.,
                                      randomizer=[False],
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
human = Human(0)
    # randomAgent = RandomAgent(1)

    # model2 = '\Agent_first_08.05_At_End.h5'
    # model1 = '\Agent_first_08.05_At_End.h5'
    # model1 = '\Agent_first_04.05_At_End.h5'
    # model2 = '\Agent_first_04.05_At_End.h5'
    # model1 = '\Agent_second__At_End.h5'
    # model2 = '\Agent_second__At_End.h5'
    # model2 = '\Agent_second__At_End.h5'

# folder = "D:\Xagents_DQN"
# model1 = '\MATRIX_first_11.05_Gross.h5'
# model2 = '\MATRIX_second_11.05_Gross.h5'
# path_model1 = folder + model1
# path_model2 = folder + model2

folder = ".\model"
first1 = '\FIX_MATRIX_first_08.06_progres.h5'
second2 = '\FIX_MATRIX_second_08.06_end.h5'
path_first1 = folder + first1
path_second2 = folder + second2

for I in range(150):
    seed1+=1


    dqn_first, dqn_second = initiateAgents()

    dqn_first.loadModel(path_first1)
    dqn_second.loadModel(path_second2)
    dqn_first.model.summary()
    dqn_second.model.summary()

    results = play_games(lambda: TicTacToeGame(), [dqn_first, dqn_second], 1500, paths = [first1, second2], plot=True, debug=True)
    print("kolejne epoki: " + str(I) + "   ----> seed1:"+ str(seed1) +" -  seed2:"+ str(seed2))
    plot_game_results(results, 2, 100, [first1, second2], " _ " + str(I))
    stats.saveStats('./stats/fixed_DDQN_10N10x3_Stats.json', seed1, first1, seed2, second2, results, 1500)
    dqn_first.saveModel(path_first1)
    dqn_second.saveModel(path_second2)
    seed2 += 1

    del dqn_first
    del dqn_second
    gc.collect()

# Agent_second_MaxReward.h5 <- słaby
# Agent_first__At_End.h5 <- dobry