
from ..game import Agent, Game
from ..tic_tac_toe import TicTacToeAction, agent_id_to_char, BOARD_DIM
import tkinter as tk

class Human(Agent):

    def __init__(self, i_agent: int):
        super().__init__(i_agent)
        self.checkbox = None
        self.root = tk.Tk()
        self.root.title("OX5")
        self.buttonsX = []
        self.checkbox = None
        self.createBoard()

    def on_button_click(self, x, y):
        try:
            self.buttonsX[x][y]["text"] = "X"
            self.action = x * BOARD_DIM + y
        except:
            print("An exception occurred")



    def createBoard(self):
        # row = []
        # buttons = []
        # button = tk.Button(self.root, text=0.0, command=self.on_button_click(1, 1))
        # button.grid(row=2, column=0)
        # row.append(button)
        # button = tk.Button(self.root, text=0.1, command=lambda: self.on_button_click(1, 1))
        # button.grid(row=1, column=0)
        # row.append(button)
        # self.checkbox = tk.Checkbutton(self.root, text="Learn", command=lambda: self.on_button_click(1, 1))
        # self.checkbox.grid(row=0, column=0)
        # row.append(self.checkbox)
        # buttons.append(row)

        for x in range(BOARD_DIM):
            row = []
            for y in range(BOARD_DIM):
                self.okVar = tk.IntVar()
                button = tk.Button(self.root, text="", width=2, height=1, font=("Helvetica", 10),
                                   command=lambda row = x, col=y: (self.on_button_click(row, col), self.okVar.set(1)))
                button.grid(row=x, column=y)
                row.append(button)
            self.buttonsX.append(row)
        # self.root.mainloop()


    def next(self, game: Game) -> bool:
        for x in range(BOARD_DIM):
            for y in range(BOARD_DIM):
                self.buttonsX[x][y]["text"] = agent_id_to_char(game.board[x*BOARD_DIM+y])
        self.root.wait_variable(self.okVar)
        action = TicTacToeAction(self.i_agent, self.action)
        return game.next(action)