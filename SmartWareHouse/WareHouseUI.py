# displsy how many win/ loss/ tie on UI
# put UI/ model into a seperate py file once have time

import os.path
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import torch.optim as optim
import numpy as np
import random
from shutil import copyfile
from tkinter import *
from tkinter import messagebox  

from Model.DL_Network_Model import Net
from Model.Gane_Agent import game_agent

# self created lib in py file format

file_name_model_latest_version = 'Model/model_latest_version.pt'

global game_result
global tie
global win
global loss
global data_original
data_original = [0]* 98
global torch_tensor
torch_tensor = torch.tensor(data_original, dtype=torch.float)

def refresh_game_board(data_original):
    for i in range(0, 98):
        if data_original[i] == -1:
            game_board[i] = 'o'
        elif data_original[i] == 1:
            game_board[i] = 'x'
        else:
            game_board[i] = ''
    for row in range(7):
        for col in range(14):
            btn_text_vars[row][col].set(game_board[row * 14 + col])
            buttons[row][col].config(bg='white',highlightbackground='white', highlightcolor='white')

def btn_reset_clicked():
    reset_game()
    refresh_game_board(data_original)

def predict_item(weight):
    model_latest_version = Net()
    state_dict_latest_version = torch.load(file_name_model_latest_version)['state_dict']
    model_latest_version.load_state_dict(state_dict_latest_version)

    # Assume input tensor should be of length 9 for the model
    input_tensor = torch.tensor([weight] * 98, dtype=torch.float)
    
    print('input_tensor:', input_tensor)
    prediction = model_latest_version(input_tensor)
    mask = torch.full_like(prediction, -float('inf'))
    if weight == 0.5:
        mask[0:27] = prediction[0:27]
    elif weight == 1:
        mask[28:69] = prediction[28:69]
    elif weight == 1.5:
        mask[70:97] = prediction[70:97]
        if all(torch_tensor[70:97] != 0):
            mask[28:69] = prediction[28:69]
    print('prediction:', prediction)
    
    for k in range(0,98):
        if torch_tensor[k] != 0:
            mask[k] = -float('inf')
    predicted_item = mask.argmax().item()
    torch_tensor[predicted_item] = 1
    print('torch_tensor:', torch_tensor)
    return predicted_item

def btn_submit_clicked():
    input_value = float(input_text.get())
    predicted_item = predict_item(input_value)
    print('predicted_item:', predicted_item)
    row = predicted_item // 14  # Integer division to find the row
    column = predicted_item % 14 
    print(f"row: {row}, column: {column}")
    cell_filled = False
    torch_tensor = torch.tensor(data_original, dtype=torch.float)
    for _ in range(98):
        if btn_text_vars[row][column].get() == '':
            btn_text_vars[row][column].set(input_value)
            if input_value == 0.5:
                color = 'green'
            elif input_value == 1:
                color = 'yellow'
            elif input_value == 1.5:
                color = 'red'
            buttons[row][column].config(bg=color,highlightbackground=color, highlightcolor=color)
            torch_tensor[predicted_item] = -1
            cell_filled = True
            break
        predicted_item = (predicted_item + 1) % 98
        row = predicted_item // 14
        column = predicted_item % 14
    torch_tensor[predicted_item] = 1
    if not cell_filled:
        print('Grid Full')
        messagebox.showinfo("Grid Full")

    return row,column


def reset_game():
    global data_original
    global game_board
    data_original = [0] * 98
    game_board = [''] * 98
    global torch_tensor
    torch_tensor = torch.tensor(data_original, dtype=torch.float)

# Load game UI:
window = Tk()
window.title("Welcome to smart warehouse")
window.geometry('1500x600')
lbl = Label(window, text="Input value:")
lbl.grid(column=16, row=2)

buttons = []
btn_text_vars = []
for row in range(7):
    btn_row = []
    var_row = []
    for col in range(14):
        btn_text = StringVar()
        button = Button(window, textvariable=btn_text, width=4, height=2, font=('Helvetica', '20'))
        button.grid(row=row, column=col)
        btn_row.append(button)
        var_row.append(btn_text)
    buttons.append(btn_row)
    btn_text_vars.append(var_row)

input_text = StringVar()
input_entry = Entry(window, textvariable=input_text)
input_entry.grid(row=3, column=16)
# button to restart
btn_reset = Button(window, text='Restart', command=btn_reset_clicked, justify=LEFT, height=2, width=8)
btn_reset.grid(row=5, column=16)
btn_submit = Button(window, text='submit', command=btn_submit_clicked, justify=LEFT, height=2, width=8)
btn_submit.grid(row=5, column=17)

# reset_game()


window.mainloop()