# from game.agents.human import Human
# from game.agents.dqn_at_last import DQNAgentAtLast
# from game.agents.dqn_at_end import DQNAgentAtEnd
from game.agents.dqn_matrix10_max_reward import DQNAgentMatrixMaxReward
from game.agents.dqn_old_end_impr import DQNAtempEnd
# from game.agents.dqn_end_matrix_end import DQNAgentEndMatrixEnd
# from game.agents.dqn_old_end_impr import DQNAtempEnd
# from game.agents.dqn_old_impr import DQNAtemp
from game.save_stats import SaveStats
# from game.agents.dqn_max_reward import DQNAgentMaxReward

# from game.agents.human import Human
from game.tic_tac_toe import TicTacToeGame
from game.utils import play_games, plot_game_results
# from game.agents import RandomAgent, DQNAgent
import gc

# agent nagradzany tylko na końcu

stats = SaveStats()
seed1, seed2 = stats.loadStats('./stats/Agent_DQN_10N10_Stats.json')

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
                                         reward_draw=50.,
                                         reward_win=100.,
                                         reward_loss=-100.,
                                         randomizer=[False],
                                         double_dqn=True,
                                         double_dqn_n_games=1,
                                         dueling_dqn=True,
                                         seed=seed1)

    dqn_second1 = DQNAtempEnd(i_agent=1,
                                      is_learning=True,
                                      learning_rate=0.001,
                                      gamma=0.95,
                                      epsilon=0.6,
                                      epsilon_end=0.0001,
                                      epsilon_decay_linear=0.0001,
                                      experience_replay_batch_size=64,
                                      pre_training_games=50,
                                      memory_size=50000,
                                      reward_draw=50.,
                                      reward_win=100.,
                                      reward_loss=-100.,
                                      randomizer=[False],
                                      double_dqn=True,
                                      double_dqn_n_games=1,
                                      dueling_dqn=True,
                                      seed=seed2)
    return dqn_first1, dqn_second1

# human = Human(0)

folder = "./models_"
first1 = '/MATRIX_first_13.05_Gross.h5'
second2 = '/MATRIX_second_13.05_Gross.h5'
path_first1 = folder + first1
path_second2 = folder + second2

for I in range(150):
    seed1+=1


    dqn_first, dqn_second = initiateAgents()

    dqn_first.loadModel(path_first1)
    dqn_second.loadModel(path_second2)
    dqn_first.model.summary()
    dqn_second.model.summary()

    results = play_games(lambda: TicTacToeGame(), [dqn_first, dqn_second], 1500, paths = [first1, second2], plot=False, debug=True)
    print("kolejne epoki: " + str(I) + "   ----> seed1:"+ str(seed1) +" -  seed2:"+ str(seed2))
    plot_game_results(results, 2, 100, [first1, second2], " _ " + str(I))
    stats.saveStats('./stats/Agent_DQN_10N10_Stats.json', seed1, first1, seed2, second2, results, 1500)
    dqn_first.saveModel(path_first1)
    dqn_second.saveModel(path_second2)
    seed2 += 1

    del dqn_first
    del dqn_second
    gc.collect()

# Agent_second_MaxReward.h5 <- słaby
# Agent_first__At_End.h5 <- dobry