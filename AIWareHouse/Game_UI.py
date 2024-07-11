import numpy as np
from tkinter import *
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import MeanSquaredError
import random as rd
import tkinter as tk
import matplotlib.pyplot as plt

# Define constants
ROWS = 7
COLS = 14

# Item types with their respective weights
ITEM_TYPES = {0: 'white', -1: 'green', 1: 'red', 2: 'beige'}
ITEM_WEIGHTS = {0: 0, 0.5: -1, 1: 1, 1.5: 2}
ITEM_COLORS = {0: 'white', -1: 'green', 1: 'red', 2: 'beige'}
COLOR_MAP = {0: (1, 1, 1), -1: (0, 1, 0), 1: (1, 0, 0), 2: (0.96, 0.96, 0.86)}

# Load the model
MODEL_PATH = 'dqn_model.h5'
mse = MeanSquaredError()
model = load_model(MODEL_PATH, custom_objects={'mse': mse})

def get_initial_frame():
    return np.zeros((ROWS, COLS), dtype=int)

def is_valid_position(frame, row, col):
    return 0 <= row < ROWS and 0 <= col < COLS and frame[row][col] == 0

def place_item_weighted(frame, weight):
    item = ITEM_WEIGHTS[weight]  # Get the item type based on weight
    if weight == 1.5:
        for row in range(ROWS-1, -1, -1):
            for col in range(COLS):
                if frame[row][col] == 0:
                    frame[row][col] = item
                    return True
    elif weight == 1:
        for row in range(ROWS-3, -1, -1):
            for col in range(COLS):
                if frame[row][col] == 0:
                    frame[row][col] = item
                    return True
    elif weight == 0.5:
        for row in range(ROWS-6, -1, -1):
            for col in range(COLS):
                if frame[row][col] == 0:
                    frame[row][col] = item
                    return True
    return False

def reshape_frame(frame):
    return np.reshape(frame, [1, ROWS * COLS])

def update_display(frame):
    for i in range(ROWS):
        for j in range(COLS):
            game_board[i][j].set(str(frame[i][j]))
            color = ITEM_COLORS.get(frame[i][j], 'green')
            buttons[i][j].config(bg=color)
            print(f"Updated cell ({i},{j}) to {frame[i][j]} with color {color}")
            window.update_idletasks()

def make_move():
    try:
        quantity = rd.randint(1, 20)
        weight = rd.choice([0.5, 1, 1.5])
        
        frame = get_initial_frame()
        state = reshape_frame(frame)

        for _ in range(quantity):
            q_values = model.predict(state, verbose=0)
            action = np.argmax(q_values[0])
            if place_item_weighted(frame, weight):
                state = reshape_frame(frame)
                update_display(frame)
            else:
                break
    except ValueError:
        print("Invalid input! Please enter integer values for quantity and weight.")

# Initialize the Tkinter window
window = tk.Tk()
window.title("Smart Warehouse")
window.geometry('900x400')

# Initialize game board buttons
game_board = [[StringVar() for _ in range(COLS)] for _ in range(ROWS)]
buttons = [[Button(window, textvariable=game_board[i][j], height=2, width=4, bg=ITEM_COLORS[0]) for j in range(COLS)] for i in range(ROWS)]

for i in range(ROWS):
    for j in range(COLS):
        buttons[i][j].grid(row=i, column=j)

# Label for user input (not used in this code, can be used for future enhancements)
label = Label(window, text="Enter something:")
label.grid(row=ROWS, column=0, columnspan=COLS)

user_input = StringVar()
entry_widget = Entry(window, textvariable=user_input)
entry_widget.grid(row=ROWS+1, column=0, columnspan=COLS)

# Button to start the moves
start_button = Button(window, text="Start", command=make_move)
start_button.grid(row=ROWS+2, columnspan=COLS)

# Start the Tkinter main loop
window.mainloop()
