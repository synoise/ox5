#
# # BOARD SIZE
from game.tic_tac_toe import BOARD_DIM
BOARD_ROW_COUNT, BOARD_COL_COUNT = BOARD_DIM, BOARD_DIM
#
# MAX_MOVE_COUNT = BOARD_ROW_COUNT * BOARD_COL_COUNT
#


EMPTY_BOARD = [[0 for c in range(BOARD_DIM)] for r in range(BOARD_DIM)]
#
# # PLAYERS
# NO_ONE = 0
HUMAN = 1
COM = -1
#
def get_opponent(player):
    if player == 1:
        return -1
    if player == -1:
        return 1
#
# # TURN
# # Human move 1st
# FIRST_TURN = HUMAN
# SECOND_TURN = COM
#
# # # COM move 1st
# # FIRST_TURN = COM
# # SECOND_TURN = HUMAN
#
# # SYMBOL
EMPTY = 0#NO_ONE
O = HUMAN
X = COM

# def agent_id_to_char(cell: GamePlayer):
#     if (cell == GamePlayer.CROSS):
#         return "X"
#     if (cell == GamePlayer.NAUGHT):
#         return "O"
#     return " "