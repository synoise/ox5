# from game.agents.human import Human
# from game.agents.dqn_at_last import DQNAgentAtLast
# from game.agents.dqn_at_end import DQNAgentAtEnd
from game.agents.dqn_matrix10_max_reward import DQNAgentMatrixMaxReward
from game.agents.dqn_end_matrix_end import DQNAgentEndMatrixEnd
from game.agents.human import Human
# from game.agents.dqn_max_reward import DQNAgentMaxReward
# from game.agents.human import Human
# from game.agents.human import Human
from game.tic_tac_toe import TicTacToeGame
from game.utils import play_games, plot_game_results
# from game.agents import RandomAgent, DQNAgent

# agent nagradzany tylko na końcu

seed1 =28
seed2 =28

def initiateAgents():
    dqn_first1 = DQNAgentMatrixMaxReward(i_agent=0,
                                         is_learning=True,
                                         learning_rate=0.0001,
                                         gamma=0.9,
                                         epsilon=0.6,
                                         epsilon_end=0.0001,
                                         epsilon_decay_linear=1 / 3000,
                                         experience_replay_batch_size=128,
                                         pre_training_games=500,
                                         memory_size=50000,
                                         reward_draw=50.,
                                         reward_win=100.,
                                         reward_loss=-100.,
                                         randomizer=[True],
                                         double_dqn=True,
                                         double_dqn_n_games=1,
                                         dueling_dqn=True,
                                         seed=seed1)

    dqn_second1 = DQNAgentEndMatrixEnd(i_agent=0,
                                      is_learning=False,
                                      learning_rate=0.001,
                                      gamma=0.9,
                                      epsilon=0.6,
                                      epsilon_end=0.0001,
                                      epsilon_decay_linear=0.0001,
                                      experience_replay_batch_size=64,
                                      pre_training_games=500,
                                      memory_size=50000,
                                      reward_draw=50.,
                                      reward_win=100.,
                                      reward_loss=-100.,
                                      randomizer=[True],
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
human = Human(1)
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

folder = "D:\Xagents_DQN"
model1 = '\MATRIX_first_13.05_Gross.h5'
model2 = '\MATRIX_second_13.05_Gross.h5'
path_model1 = folder + model1
path_model2 = folder + model2

for I in range(150):
    seed1+=1


    dqn_first, dqn_second = initiateAgents()

    dqn_first.loadModel(path_model1)
    dqn_second.loadModel(path_model2)
    dqn_first.model.summary()
    dqn_second.model.summary()

    results = play_games(lambda: TicTacToeGame(), [dqn_second, human], 1800, paths = [model1, model2], plot=True, debug=True)
    print("kolejne epoki: " + str(I) + "   ----> seed1:"+ str(seed1) +" -  seed2:"+ str(seed2))
    plot_game_results(results, 2, 100, [model1, model2], " _ " + str(I))

    dqn_first.saveModel(path_model1)
    dqn_second.saveModel(path_model2)

    del dqn_first
    del dqn_second

    seed2 += 1

# Agent_second_MaxReward.h5 <- słaby
# Agent_first__At_End.h5 <- dobry
import json

# def saveStats(fileJSON,seed1,seed2,file1,file2):
#     data = {
#         'file_1': file1,
#         'seed_1': seed1,
#         'file_2': file2,
#         'seed_2': seed2,
#     }
#
#     # Otwórz plik JSON w trybie zapisu
#     with open(fileJSON, 'w') as jsonfile:
#         # Zapisz dane do pliku w formacie JSON
#         json.dump(data, jsonfile)
#
# def loadStats(fileJSON):
#     with open(fileJSON, 'r') as jsonfile:
#         data = json.load(jsonfile)
#
#     print(data['seed_1'])
#     print(data['seed_2'])
#
# # saveStats('stats.json',1,2,3,4)
# # saveStats('Agent_DQN_Stats.json', 1, 3, 4, 5)
# loadStats('Agent_DQN_Stats.json')