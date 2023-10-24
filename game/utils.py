import datetime
from typing import List, Callable
import matplotlib
import matplotlib.pyplot as plt
from cycler import cycler
from collections import Counter
from IPython import display
# from .boardox import BoardOX
from .game import Game, Agent
from .tic_tac_toe import agent_id_to_char, GamePlayer

agent_signs: list[GamePlayer] = [GamePlayer.NAUGHT, GamePlayer.CROSS]
def play_game(game: Game, agents: List[Agent]) -> List[int]:
    for agent in agents:
        agent.new_game(game)

    while not agents[game.get_current_agent()].next(game):
        # print(game.__str__())
        pass

    for agent in agents:
        agent.end_game(game)

    # print("first",agents[0])
    print(game.__str__())

    return game.get_winners()



def play_games(create_game: Callable[[], Game], agents: List[Agent],
               n_games: int = 10000,
               paths:List=[],
               debug: bool = False,
               plot: bool = False, plot_window: int = 20, plot_update_n_games: int = 300) -> List[int]:
    results = []

    for i in range(n_games):
        game = create_game()


        # if i % 2:
        winners = play_game(game, [agents[0],agents[1]])
        # else:
        #     winners = play_game(game,[agents[1],agents[0]])

        if len(winners) > 1:
            results.append(-1)
        else:
            results.append(winners[0])

        print("_________________________________________________    epoka:", i,", winer",  agent_id_to_char(agent_signs[winners[0]]), " :: ", winners)
        # border.on_button_click(i,3)
        if plot and ((i + 1) % plot_update_n_games == 0 or i == n_games - 1):
            display.clear_output(wait=True)
            plot_game_results(results, len(agents), plot_window,paths)
            display.display(plt.gcf())
            plt.clf()

    if debug:
        counts = Counter(results)
        print(
            "Po {} grach mamy - remisy: {}, wins: {}.".format(
                n_games,
                "{} ({:.2%})".format(counts[-1], counts[-1] / n_games),
                ", ".join(["{} ({:.2%})".format(counts[i], counts[i] / n_games) for i in range(len(agents))])
            )
        )

    return results


def moving_count(items: List[int], value: int, window: int) -> List[int]:
    count = 0
    results = []
    for i in range(len(items)):
        count += -1 if i - window >= 0 and items[i - window] == value else 0
        count += 1 if items[i] == value else 0
        if i >= window - 1:
            results.append(count / window)
    return results


def plot_game_results(results: List[int], num_agents: int, window: int = 100,paths=["none","none"],tura="",colors = ['r', 'g', 'b']):
    game_number = range(window, len(results) + 1)
    draws = moving_count(results, -1, window)
    winners = [moving_count(results, i, window) for i in range(num_agents)]

    # if colors:
    plt.rc('axes', prop_cycle=(cycler('color', colors)))
    # else:
    #     plt.rc('axes', prop_cycle=(cycler('color', ['b', 'purple', 'orange'])))

    plt.plot(game_number, draws, label='Remis')
    for i, winner in enumerate(winners, start=0):
        plt.plot(game_number, winner, label= 'Agent ' + str(i) + " :: " + agent_id_to_char(agent_signs[i]) + ' wins'+paths[i])
    plt.ylabel(f'Ocena w końcowym oknie {window} | ')
    teraz = datetime.datetime.now()
    print(paths)
    plt.xlabel(' Epoki: ' + tura + "| Czas: " + teraz.strftime("%H:%M") + "  " + teraz.strftime("%d-%m") )
    plt.xlim([0, len(results)])
    plt.ylim([0, 1])
    ax = plt.gca()
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(xmax=1))
    ax.xaxis.set_major_formatter(matplotlib.ticker.StrMethodFormatter('{x:,.0f}'))
    plt.legend(loc='best')
    # plt.plot([1, 2, 3, 4])
    # plt.ylabel('some numbers')
    plt.show()