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

from Model.DL_Network_Model import Net
from Model.Gane_Agent import game_agent

# self created lib in py file format

file_name_model_latest_version = 'Model/model_latest_version.pt'
file_name_model_last_version = 'Model/model_last_version.pt'

global game_result
global tie
global win
global loss


def trigger_AI(data_original):
    game_result = "Ongoing.."
    torch_tensor = torch.tensor(data_original, dtype=torch.float)

    # another possible scenario: if input has worong occurance of 1 & -1, then it is not valid input as well - pending for coding
    game_a = game_agent(torch_tensor, 0)
    if game_a.verify_result() == True:
        print('Win')
        game_result = "Win"
        next_step = -1
    elif 0 not in torch_tensor:
        print('tie')
        game_result = "tie"
        next_step = -1
    else:
        # here it can be 3 levels of AI for selection
        if AI_difficulty == 'H':
            next_vision = model_latest_version(torch_tensor)
        else:
            next_vision = model_last_version(torch_tensor)
        # comprehance predict result to make impossible option as '-1'
        for k in range(0, 9):
            if torch_tensor[k] != 0:
                next_vision[k] = -1.1
        next_step = next_vision.argmax()
        print(next_vision)
        torch_tensor[next_step] = 1
        game_a = game_agent(torch_tensor, 0)
        # varify status of result
        if game_a.verify_result() == True:
            game_result = 'Loss'
        elif 0 not in torch_tensor:
            print('tie')
            game_result = 'tie'
            # tie = tie + 1
    return next_step, game_result


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


def button_00_clicked():
    # lbl.configure(text="Button was clicked !!")
    if btn_00_text.get() == '':
        btn_00_text.set("o")
        data_original[0] = -1
        print(data_original)
        next_step, game_result = trigger_AI(data_original)
        print(next_step)
        if next_step != -1:
            data_original[next_step] = 1
        refresh_game_board(data_original)
        lbl.configure(text=game_result)


def button_01_clicked():
    # lbl.configure(text="Button was clicked !!")
    if btn_01_text.get() == '':
        btn_01_text.set("o")
        data_original[1] = -1
        print(data_original)
        next_step, game_result = trigger_AI(data_original)
        print(next_step)
        if next_step != -1:
            data_original[next_step] = 1
        refresh_game_board(data_original)
        lbl.configure(text=game_result)


def button_02_clicked():
    # lbl.configure(text="Button was clicked !!")
    if btn_02_text.get() == '':
        btn_02_text.set("o")
        data_original[2] = -1
        print(data_original)
        next_step, game_result = trigger_AI(data_original)
        print(next_step)
        if next_step != -1:
            data_original[next_step] = 1
        refresh_game_board(data_original)
        lbl.configure(text=game_result)


def button_10_clicked():
    # lbl.configure(text="Button was clicked !!")
    if btn_10_text.get() == '':
        btn_10_text.set("o")
        data_original[3] = -1
        print(data_original)
        next_step, game_result = trigger_AI(data_original)
        print(next_step)
        if next_step != -1:
            data_original[next_step] = 1
        refresh_game_board(data_original)
        lbl.configure(text=game_result)


def button_11_clicked():
    # lbl.configure(text="Button was clicked !!")
    if btn_11_text.get() == '':
        btn_11_text.set("o")
        data_original[4] = -1
        print(data_original)
        next_step, game_result = trigger_AI(data_original)
        print(next_step)
        if next_step != -1:
            data_original[next_step] = 1
        refresh_game_board(data_original)
        lbl.configure(text=game_result)


def button_12_clicked():
    # lbl.configure(text="Button was clicked !!")
    if btn_12_text.get() == '':
        btn_12_text.set("o")
        data_original[5] = -1
        print(data_original)
        next_step, game_result = trigger_AI(data_original)
        print(next_step)
        if next_step != -1:
            data_original[next_step] = 1
        refresh_game_board(data_original)
        lbl.configure(text=game_result)


def button_20_clicked():
    # lbl.configure(text="Button was clicked !!")
    if btn_20_text.get() == '':
        btn_20_text.set("o")
        data_original[6] = -1
        print(data_original)
        next_step, game_result = trigger_AI(data_original)
        print(next_step)
        if next_step != -1:
            data_original[next_step] = 1
        refresh_game_board(data_original)
        lbl.configure(text=game_result)


def button_21_clicked():
    # lbl.configure(text="Button was clicked !!")
    if btn_21_text.get() == '':
        btn_21_text.set("o")
        data_original[7] = -1
        print(data_original)
        next_step, game_result = trigger_AI(data_original)
        print(next_step)
        if next_step != -1:
            data_original[next_step] = 1
        refresh_game_board(data_original)
        lbl.configure(text=game_result)


def button_22_clicked():
    # lbl.configure(text="Button was clicked !!")
    if btn_22_text.get() == '':
        btn_22_text.set("o")
        data_original[8] = -1
        print(data_original)
        next_step, game_result = trigger_AI(data_original)
        print(next_step)
        if next_step != -1:
            data_original[next_step] = 1
        refresh_game_board(data_original)
        lbl.configure(text=game_result)


def btn_reset_clicked():
    reset_game()
    refresh_game_board(data_original)


def model_select():
    global model_selection
    if AI_difficulty == 'L':
        model_selection = state_dict_last_version
    else:
        model_selection = file_name_model_latest_version


def reset_game():
    global data_original
    global game_board
    data_original = [0, 0, 0, 0, 0, 0, 0, 0, 0]
    game_board = ['', '', '', '', '', '', '', '', '']

    if (AI_first == 1) & (AI_random_first_step == 1):
        data_original[random.randint(0, 8)] = 1
    elif (AI_first == 1):
        torch_tensor = torch.tensor(data_original, dtype=torch.float)
        if AI_difficulty == 'H':
            next_vision = model_latest_version(torch_tensor)
        else:
            next_vision = model_last_version(torch_tensor)
        # comprehance predict result to make impossible option as '-1.1'
        for k in range(0, 9):
            if torch_tensor[k] != 0:
                next_vision[k] = -1.1
        next_step = next_vision.argmax()
        data_original[next_step] = 1


def AI_difficulty_selection():
    global AI_random_first_step
    global AI_difficulty
    if AI_difficulty_var.get() == 'H':
        AI_random_first_step = 0
        AI_difficulty = 'H'
    elif AI_difficulty_var.get() == 'M':
        AI_random_first_step = 1
        AI_difficulty = 'M'
    elif AI_difficulty_var.get() == 'L':
        AI_random_first_step = 1
        AI_difficulty = 'L'
    print(AI_difficulty_var.get())
    reset_game()
    refresh_game_board(data_original)


def Who_play_first_selection():
    global AI_first
    print(Who_play_first_var.get())
    if Who_play_first_var.get() == 'AI':
        AI_first = 1
    else:
        AI_first = 0
    reset_game()
    refresh_game_board(data_original)



AI_first = 1
# AI_random_first_step = 0
tie = 0
win = 0
loss = 0

# load model
model_last_version = Net()
state_dict_last_version = torch.load(file_name_model_last_version)['state_dict']
model_last_version.load_state_dict(state_dict_last_version)
model_latest_version = Net()
state_dict_latest_version = torch.load(file_name_model_latest_version)['state_dict']
model_latest_version.load_state_dict(state_dict_latest_version)

# Load game UI:
window = Tk()
window.title("Welcome to Game made by Hongxu")
window.geometry('450x200')
lbl = Label(window, text="Ready to start")

btn_00_text = StringVar()
btn_01_text = StringVar()
btn_02_text = StringVar()
btn_10_text = StringVar()
btn_11_text = StringVar()
btn_12_text = StringVar()
btn_20_text = StringVar()
btn_21_text = StringVar()
btn_22_text = StringVar()

btn_00 = Button(window, textvariable=btn_00_text, command=button_00_clicked, height=2, width=4)
btn_00.grid(row=0, column=0)
btn_01 = Button(window, textvariable=btn_01_text, command=button_01_clicked, height=2, width=4)
btn_01.grid(row=0, column=1)
btn_02 = Button(window, textvariable=btn_02_text, command=button_02_clicked, height=2, width=4)
btn_02.grid(row=0, column=2)
btn_10 = Button(window, textvariable=btn_10_text, command=button_10_clicked, height=2, width=4)
btn_10.grid(row=1, column=0)
btn_11 = Button(window, textvariable=btn_11_text, command=button_11_clicked, height=2, width=4)
btn_11.grid(row=1, column=1)
btn_12 = Button(window, textvariable=btn_12_text, command=button_12_clicked, height=2, width=4)
btn_12.grid(row=1, column=2)
btn_20 = Button(window, textvariable=btn_20_text, command=button_20_clicked, height=2, width=4)
btn_20.grid(row=2, column=0)
btn_21 = Button(window, textvariable=btn_21_text, command=button_21_clicked, height=2, width=4)
btn_21.grid(row=2, column=1)
btn_22 = Button(window, textvariable=btn_22_text, command=button_22_clicked, height=2, width=4)
btn_22.grid(row=2, column=2)
lbl.grid(row=4, column=3)

AI_difficulty_lbl = Label(window, text="AI difficulty:")
AI_difficulty_lbl.grid(row=1, column=3)
AI_difficulty_var = StringVar()
AI_difficulty_r1 = Radiobutton(window, text='High', variable=AI_difficulty_var, value='H', justify=LEFT,
                               command=AI_difficulty_selection)
AI_difficulty_r2 = Radiobutton(window, text='Medium', variable=AI_difficulty_var, value='M', justify=LEFT,
                               command=AI_difficulty_selection)
AI_difficulty_r3 = Radiobutton(window, text='Low', variable=AI_difficulty_var, value='L', justify=LEFT,
                               command=AI_difficulty_selection)
AI_difficulty_var.set("H")
AI_difficulty_selection()
AI_difficulty_r1.grid(row=1, column=5)
AI_difficulty_r2.grid(row=1, column=6)
AI_difficulty_r3.grid(row=1, column=7)

Who_play_first_lbl = Label(window, text="Who play first?")
Who_play_first_lbl.grid(row=0, column=3)
Who_play_first_var = StringVar()
Who_play_first_r1 = Radiobutton(window, text='Player', variable=Who_play_first_var, value='Player', justify=LEFT,
                                command=Who_play_first_selection)
Who_play_first_r2 = Radiobutton(window, text='AI', variable=Who_play_first_var, value='AI', justify=LEFT,
                                command=Who_play_first_selection)
Who_play_first_var.set("Player")
Who_play_first_r1.grid(row=0, column=5)
Who_play_first_r2.grid(row=0, column=6)
Who_play_first_selection()

# button to restart
btn_reset = Button(window, text='Restart', command=btn_reset_clicked, justify=LEFT, height=2, width=8)
btn_reset.grid(row=2, column=5)

reset_game()
# refresh_game_board(data_original)

window.mainloop()