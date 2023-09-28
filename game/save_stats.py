import datetime
import json

import matplotlib
from cycler import cycler
from matplotlib import pyplot as plt


class SaveStats:
    def __init__(self):
        self.colors = ['r', 'g', 'b']
        self.first_wins = []
        self.second_wins = []
        self.remis = []
        self.time = []
        self.epok = 0

    def resulting(self, results):
        self.first_wins.append(results.count(0))
        self.second_wins.append(results.count(1))
        self.remis.append(results.count(-1))
        return self.first_wins, self.second_wins, self.remis

    def saveStats(self, fileJSON, seed1, file1, seed2, file2, results, n):
        wins_1st, wins_2nd, rem = self.resulting(results)
        self.epok = self.epok + n
        teraz = datetime.datetime.now()
        self.time.append(teraz.strftime("%H:%M") + "  " + teraz.strftime("%d-%m"),)

        data = {
            'first_file': file1,
            'first_seed': seed1,
            'first_wins': wins_1st,
            'second_file': file2,
            'second_seed': seed2,
            'second_wins': wins_2nd,
            'remis': rem,
            'time': self.time,
            'epok': (self.epok),
            'colors': self.colors
        }

        with open(fileJSON, 'w') as jsonfile:
            json.dump(data, jsonfile)

    def loadStats(self, fileJSON):
        with open(fileJSON, 'r') as jsonfile:
            data = json.load(jsonfile)

        self.first_wins = data['first_wins']
        self.second_wins = data['second_wins']
        self.remis = data['remis']
        self.epok = data['epok']
        self.colors = data['colors']
        self.time = data['time']
        self.first_file = data['first_file']
        self.second_file = data['second_file']
        return data['first_seed'], data['second_seed']

    def printStats(self, param):
        self.loadStats(param)
        # game_number = range(0, self.epok, 1500)
        game_number = range(0, len(self.first_wins))
        plt.rc('axes', prop_cycle=(cycler('color',  self.colors)))

        plt.plot(game_number, self.remis, label= "remis :: " + str(sum(self.remis)))
        # for i, winner in enumerate(winners, start=0):
        plt.plot(game_number, self.first_wins, label= self.first_file + " :: " + str(sum(self.first_wins)))
        plt.plot(game_number, self.second_wins, label= self.second_file + " :: " + str(sum(self.second_wins)))
        window=33
        plt.ylabel(f'Ocena w końcowym oknie {window} | ')

        # def add_1500(value, pos):
        #     return '{:,.0f}'.format(value + 1500)

        plt.xlabel(' Epoki: '+ str(self.epok) )
        # plt.xlim([0, 1]) #len(results)])
        # plt.ylim([0, 1])
        ax = plt.gca()
        # ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(xmax=1))
        ax.xaxis.set_major_formatter(matplotlib.ticker.StrMethodFormatter('{x:,.0f}'))
        # ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(1))
        plt.legend(loc='best')
        plt.show()
        pass


# ss = SaveStats()
# ss.printStats(param='..\Agent_DQN_Stats.json')
# ss.printStats(param='..\stats\Agent_DQN_10N10_Stats.json')
# ss.printStats(param='..\Fixed_DDQN_10N10x3_Stats.json')
