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
data_original = [0, 0, 0, 0, 0, 0, 0, 0, 0]
global torch_tensor
torch_tensor = torch.tensor(data_original, dtype=torch.float)

def refresh_game_board(data_original):
    for i in range(0, 9):
        if data_original[i] == -1:
            game_board[i] = 'o'
        elif data_original[i] == 1:
            game_board[i] = 'x'
        else:
            game_board[i] = ''
    btn_00_text.set(game_board[0])
    btn_01_text.set(game_board[1])
    btn_02_text.set(game_board[2])
    btn_10_text.set(game_board[3])
    btn_11_text.set(game_board[4])
    btn_12_text.set(game_board[5])
    btn_20_text.set(game_board[6])
    btn_21_text.set(game_board[7])
    btn_22_text.set(game_board[8])

def btn_reset_clicked():
    reset_game()
    refresh_game_board(data_original)

def predict_item(weight):
    model_latest_version = Net()
    state_dict_latest_version = torch.load(file_name_model_latest_version)['state_dict']
    model_latest_version.load_state_dict(state_dict_latest_version)

    # Assume input tensor should be of length 9 for the model
    input_tensor = torch.tensor([weight] * 9, dtype=torch.float)
    
    print('input_tensor:', input_tensor)
    prediction = model_latest_version(input_tensor)
    mask = torch.full_like(prediction, -float('inf'))
    if weight == 0.5:
        mask[0:3] = prediction[0:3]
    elif weight == 1:
        mask[3:6] = prediction[3:6]
    elif weight == 1.5:
        mask[6:9] = prediction[6:9]
    print('prediction:', prediction)
    
    for k in range(0,9):
        if torch_tensor[k] != 0:
            mask[k] = -1.1
    predicted_item = mask.argmax().item()
    torch_tensor[predicted_item] = 1
    print('torch_tensor:', torch_tensor)
    return predicted_item
def btn_submit_clicked():
    input_value = float(input_text.get())
    predicted_item = predict_item(input_value)
    print('predicted_item:', predicted_item)
    row = predicted_item // 3  # Integer division to find the row
    column = predicted_item % 3 
    print(row,column)
    cell_filled = False
    torch_tensor = torch.tensor(data_original, dtype=torch.float)
    for _ in range(9):
        if btn_text_vars[row][column].get() == '':
            btn_text_vars[row][column].set(input_value)
            torch_tensor[predicted_item] = -1
            cell_filled = True
            break
        predicted_item = (predicted_item + 1) % 9
        row = predicted_item // 3
        column = predicted_item % 3
    torch_tensor[predicted_item] = 1
    if not cell_filled:
        print('Grid Full')
        messagebox.showinfo("Grid Full")

    return row,column


def reset_game():
    global data_original
    global game_board
    data_original = [0, 0, 0, 0, 0, 0, 0, 0, 0]
    game_board = ['', '', '', '', '', '', '', '', '']
    global torch_tensor
    torch_tensor = torch.tensor(data_original, dtype=torch.float)

# Load game UI:
window = Tk()
window.title("Welcome to smart warehouse")
window.geometry('700x600')
lbl = Label(window, text="Input value:")
lbl.grid(column=1, row=8)

btn_00_text = StringVar()
btn_01_text = StringVar()
btn_02_text = StringVar()
btn_10_text = StringVar()
btn_11_text = StringVar()
btn_12_text = StringVar()
btn_20_text = StringVar()
btn_21_text = StringVar()
btn_22_text = StringVar()

btn_text_vars = [[btn_00_text, btn_01_text, btn_02_text],
                 [btn_10_text, btn_11_text, btn_12_text],
                 [btn_20_text, btn_21_text, btn_22_text]]

btn_00 = Button(window, textvariable=btn_00_text,  height=2, width=4)
btn_00.grid(row=0, column=0)
btn_01 = Button(window, textvariable=btn_01_text,  height=2, width=4)
btn_01.grid(row=0, column=1)
btn_02 = Button(window, textvariable=btn_02_text,  height=2, width=4)
btn_02.grid(row=0, column=2)
btn_10 = Button(window, textvariable=btn_10_text,  height=2, width=4)
btn_10.grid(row=1, column=0)
btn_11 = Button(window, textvariable=btn_11_text,  height=2, width=4)
btn_11.grid(row=1, column=1)
btn_12 = Button(window, textvariable=btn_12_text,  height=2, width=4)
btn_12.grid(row=1, column=2)
btn_20 = Button(window, textvariable=btn_20_text,  height=2, width=4)
btn_20.grid(row=2, column=0)
btn_21 = Button(window, textvariable=btn_21_text,  height=2, width=4)
btn_21.grid(row=2, column=1)
btn_22 = Button(window, textvariable=btn_22_text, height=2, width=4)
btn_22.grid(row=2, column=2)
lbl.grid(row=4, column=3)


input_text = StringVar()
input_entry = Entry(window, textvariable=input_text)
input_entry.grid(row=8, column=3)
# button to restart
btn_reset = Button(window, text='Restart', command=btn_reset_clicked, justify=LEFT, height=2, width=8)
btn_reset.grid(row=8, column=5)
btn_submit = Button(window, text='submit', command=btn_submit_clicked, justify=LEFT, height=2, width=8)
btn_submit.grid(row=8, column=4)

reset_game()


window.mainloop()