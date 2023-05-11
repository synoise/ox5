import random
BOARD_SIZE = 10
# BOARD_DIM = BOARD_DIM
tableSize = BOARD_SIZE ** 2




def checkLines(boardDimension):
    tab = []
    # CHECK_LINES = [
    for i in range(boardDimension):
        for j in range(boardDimension):
            if j < boardDimension - 4:
                tab.append([j + i * boardDimension, 1 + j + i * boardDimension, 2 + j + i * boardDimension, 3 + j + i * boardDimension,
                            4 + j + i * boardDimension])  # rows

            d = i * boardDimension + j

            if i < boardDimension - 4 and j < boardDimension - 4:
                tab.append([d, d + boardDimension + 1, d + boardDimension * 2 + 2, d + boardDimension * 3 + 3,
                            d + boardDimension * 4 + 4])  # rows
            d = i * boardDimension + j

            if 3 < i < boardDimension and j < boardDimension - 4:
                tab.append([d, d - boardDimension + 1, d - boardDimension * 2 + 2, d - boardDimension * 3 + 3,
                            d - boardDimension * 4 + 4])  # rows

    for i in range(boardDimension ** 2 - boardDimension * 4):
        tab.append([i, i + boardDimension, i + boardDimension * 2, i + boardDimension * 3, i + boardDimension * 4])  # cols

    # if j < boardDimension - 5 and i < boardDimension - 5:
    #     tab.append([i * j, i * j + boardDimension + 1, i * j + boardDimension * 2 + 2, i * j + boardDimension * 3 + 3, i * j + boardDimension * 4 + 4])  # diagonals
    # if j < boardDimension - 5 and i + 5 < boardDimension:
    #     tab.append([i * j + 4, i * j + boardDimension + 3, i * j + boardDimension * 2 + 2, i * j + boardDimension * 3 + 1, i * j + boardDimension * 4])  #
    return tab




CHECK_LINES = checkLines(BOARD_SIZE)

def find_max_tuple(cells):
    try:
        max_value = max(cell[1] for row in cells for cell in row if isinstance(cell, tuple))
        x = [cell for row in cells for cell in row if isinstance(cell, tuple) and cell[1] == max_value]
        x =random.choice(x)
        return (x[0], 0)   #max_value
    except:
        return []

def mielimy(tab, start, stop, step, agent, cell, xxx):
    arr = tab[start:stop:step]
    if len(arr) > 4:
        pass

    for i in range(start, stop, step):
        if i == cell:
            continue
        if (i < tableSize and tab[i] == 0):  # and len(arr) > 4 ):

            return i, agent * sum(arr)
    pass


def addNear(tab, start, stop, step, agent, cell, xxx):
    # print(agent,tab[start: stop: step])
    for i in range(start, stop, step):
        if i == cell:
            continue
        if tab[i] == 0:
            return i, 0
    pass


def getAwardTrzy(tab, start, stopMax, stopMin, step, agent, cell, xxx):
    # print(tab[stopMin:stopMax:step],start, stopMin, stopMax, step,xxx)
    moves = []  # start,start in tab[stopMin:stopMax:step], tab[stopMin:stopMax:step],xxx
    if not - agent in tab[start:stopMax:step][1:4] and not - agent in tab[start:stopMin:-step][1:4]:
        moves.append(mielimy(tab, stopMin, stopMax, step, agent, cell, xxx))
        # m oves.append(mielimy(tab, start, stopMin, -step, agent, cell, xxx))
    if 0 in tab[start:stopMax:step]:
        moves.append(addNear(tab, start, min(stopMax, 99), step, agent, cell, xxx))
        moves.append(addNear(tab, start, stopMin, step, agent, cell, xxx))
    else:
        pass
    return moves


def addMoves(move, tab):
    if tab:
        move.append(tab)
    return move
    pass


def searchForDangerousAndWin(agent, tab):

    for indexes in CHECK_LINES:
        line = [tab[i] for i in indexes]
        # moves = [[val for val in sublst if val is not None] for sublst in moves]
        if [-agent, 0, -agent, -agent, -agent] == line:
            return indexes[1]
        elif [-agent, -agent, 0, -agent, -agent] == line:
            return indexes[2]
        elif [-agent, -agent, -agent, 0, -agent] == line:
            return indexes[3]
        elif [-agent, -agent, -agent, -agent, 0] == line:
            return indexes[-1]
        elif [0, -agent, -agent, -agent, -agent] == line:
            return indexes[0]

        # elif [0, -agent, -agent, -agent, 0] == line:
        #     return random.choice([indexes[0], indexes[-1]])
        # if sum(line) == -4:
        #     print(line,indexes)
        #     for i in indexes:
        #         if tab[i] == 0:
        #             return i
    return False


def searchForDangerousAndWin2(agent, tab):
    for indexes in CHECK_LINES:
        line = [tab[i] for i in indexes]
        if [0, -agent, -agent, -agent, 0] == line:
            return random.choice([indexes[0], indexes[-1]])
    return False

def getCloestElement(cell, tab, agent):

    result = searchForDangerousAndWin(-agent, tab) # can win !
    if result:
        return (result,4)

    result = searchForDangerousAndWin(agent, tab) # can loss !
    if result:
        return (result,4)

    result = searchForDangerousAndWin2(-agent, tab) # can win
    if result:
        return (result, 2)

    result = searchForDangerousAndWin2(agent, tab) # can loss
    if result:
        return (result, 2)

    # result = searchForDangerous(agent, tab, [-agent, -agent, -agent, -agent,0])
    # if result:
    #     return [(result,10)]
    # print("B")
    # result = searchForDangerous(agent, tab, [0,-agent, -agent, -agent, -agent])
    # if result:
    #     return [(result,10)]
    #
    # print("c")


    row = cell // BOARD_SIZE
    col = cell % BOARD_SIZE

    moves = []

    rowx10 = BOARD_SIZE * row
    xxx = (row + col) % BOARD_SIZE

    if (row + col < BOARD_SIZE):
        c = xxx
        d = xxx * BOARD_SIZE + 1
    else:
        c = xxx * BOARD_SIZE + 2 * BOARD_SIZE - 1
        d = tableSize

    b = max(col - row, 0) * BOARD_SIZE + abs(min(col - row, 0)) * BOARD_SIZE
    a = rowx10 + BOARD_SIZE * (BOARD_SIZE - col)

    addMoves(moves, getAwardTrzy(tab, cell, d, c, BOARD_SIZE - 1, agent, cell, "skos /"))
    addMoves(moves, getAwardTrzy(tab, cell, a, b, BOARD_SIZE + 1, agent, cell, "skos \ "))
    addMoves(moves, getAwardTrzy(tab, cell, rowx10 + BOARD_SIZE, rowx10, 1, agent, cell, "poziom -"))
    moves = addMoves(moves, getAwardTrzy(tab, cell, tableSize + col, col, BOARD_SIZE, agent, cell, "pion |"))

    moves =[[val for val in sublst if val is not None] for sublst in moves]
    return find_max_tuple(moves)

# random.seed(47)
# mozliwe_wartosci = [-1,0, 1]
# tablica = [random.choice(mozliwe_wartosci) for _ in range(100)]
# # tablica = list(range(100))
# #
# #
# posi = 82
# # tablica[posi] = 66
# #
# for i in range(0, 100, 10):
#     print(tablica[i:i + 10])
#
# print(getCloestElement(posi, tablica, 1))
