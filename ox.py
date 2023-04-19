import tkinter as tk
from tkinter import messagebox


# Funkcja do obsługi kliknięcia na przycisk
def on_button_click(buttonxxx):
    print(buttonxxx)
    if buttonxxx["text"] == "":
        buttonxxx["text"] = current_player
        check_win()
        switch_player()


# Funkcja do sprawdzenia, czy gra została wygrana
def check_win():
    winning_combinations = [
        [(0, 0), (0, 1), (0, 2)],
        [(1, 0), (1, 1), (1, 2)],
        [(2, 0), (2, 1), (2, 2)],
        [(0, 0), (1, 0), (2, 0)],
        [(0, 1), (1, 1), (2, 1)],
        [(0, 2), (1, 2), (2, 2)],
        [(0, 0), (1, 1), (2, 2)],
        [(0, 2), (1, 1), (2, 0)]
    ]

    for combination in winning_combinations:
        a = buttons[combination[0][0]][combination[0][1]]["text"]
        b = buttons[combination[1][0]][combination[1][1]]["text"]
        c = buttons[combination[2][0]][combination[2][1]]["text"]
        if a == b and b == c and a != "":
            messagebox.showinfo("Wygrana", f"Gracz {a} wygrywa!")
            restart_game()


# Funkcja do zmiany gracza
def switch_player():
    global current_player
    if current_player == "X":
        current_player = "O"
    else:
        current_player = "X"


# Funkcja do restartu gry
def restart_game():
    global current_player
    current_player = "X"
    for i in range(3):
        for j in range(3):
            buttons[i][j]["text"] = ""


# Tworzenie okna
root = tk.Tk()
root.title("Tic-Tac-Toe")

# Tworzenie przyciskówaz
buttons = []
for i in range(3):
    row = []
    for j in range(3):
        button = tk.Button(root, text="", width=10, height=3, font=("Helvetica", 24),
                           command=lambda row=i, col=j: on_button_click(buttons[row][col]))
        button.grid(row=i, column=j)
        row.append(button)
    buttons.append(row)

# Zmienna przechowująca aktualnego gracza
current_player = "X"

# Uruchomienie pętli głównej
root.mainloop()
