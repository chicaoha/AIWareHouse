import os.path
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import random as rd
from shutil import copyfile

from Model.DL_Network_Model import Net
from Model.Function_Bank import data_input
from Model.Gane_Agent import game_agent

file_name_model_latest_version = 'Model/model_latest_version.pt'
file_name_model_last_version = 'Model/model_last_version.pt'

input_file_path = 'Data/training_data_input.csv'
score_file_path = 'Data/training_data_score.csv'
training_file_path = 'Data/training_data.csv'
next_action_file_path = 'Data/training_data_next_action_taken.csv'

# Data format setup
torch.set_default_dtype(torch.float32)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Parameter setup
button = 1
train_model_from_crash = 0
main_loop_count = 5
epoch_size = 3000
steps_for_printing_out_loss = 1000
learning_rate = 0.2
DQ_ratio = 0.75 
weights = [0.5, 1, 1.5]

def Training_model():

    indicator = 0
    input_data = data_input(input_file_path).to(device)
    score_data = data_input(score_file_path).to(device)
    

    input = input_data
    target = score_data
    net = Net().to(device)
    loss_function = nn.MSELoss()
    optimizer = optim.SGD(net.parameters(), lr=learning_rate)

    if indicator == 1:
        state_dict_last_version = torch.load(file_name_model_latest_version)['state_dict']
        net.load_state_dict(state_dict_last_version)

    for i in range(1, epoch_size + 1):
        optimizer.zero_grad()
        output = net(input)
        output[target == -2] = -2
        output[target == -1.1] = -1.1
        loss = loss_function(output, target)
        loss.backward()
        if i % steps_for_printing_out_loss == 0:
            print(f'Loss (epoch: {i}): {loss.cpu().detach().numpy()}')
        optimizer.step()

    if indicator == 1:
        copyfile(file_name_model_latest_version, file_name_model_last_version)
    else:
        torch.save({'state_dict': net.state_dict(), 'optimizer': optimizer.state_dict()}, file_name_model_last_version)
    torch.save({'state_dict': net.state_dict(), 'optimizer': optimizer.state_dict()}, file_name_model_latest_version)

    export_output = output.clone().cpu().detach().numpy()
    np.savetxt('Data/output.csv', export_output, delimiter=',')
    export_target = target.clone().cpu().detach().numpy()
    np.savetxt('Data/target.csv', export_target, delimiter=',')


def RL_model():
    win = 0
    loss = 0
    model_latest_version = Net().to(device)
    state_dict_latest_version = torch.load(file_name_model_latest_version)['state_dict']
    model_latest_version.load_state_dict(state_dict_latest_version)

    training_data = data_input(training_file_path).to(device)

    for model_sequence in range(2):
        for x in range(82):
            torch_tensor = training_data[x].clone()
            input_status = []
            next_action_taken = []
            score = []

            for j in range(98):
                if j == model_sequence:
                    game_a = game_agent(torch_tensor, 0)
                   

                weight = weights[(model_sequence +j) % 3] 
                # weight = rd.choice(weights) 
                torch_tensor_saved = torch_tensor.clone()
                input_status.append(torch_tensor_saved.cpu().numpy())

                
                next_vision = model_latest_version(torch_tensor)
               

                current_score = torch.ones(98, device=device) * -2
                for k in range(98):
                    if torch_tensor[k] != 0:
                        next_vision[k] = -1.1
                        # current_score[k] = -1.1
                next_step = next_vision.argmax()

                next_action_taken.append(next_step.cpu().numpy())
                # score.append(current_score.cpu().numpy())

                torch_tensor[next_step] = 1
                game_a = game_agent(torch_tensor, 0)
                result = game_a.verify_result_weight(next_vision, weight, next_step)
                if result:
                    current_score[next_step] = 1.1
                    score.append(current_score.cpu().numpy())
                    win += 1
                else:
                    current_score[next_step] = -2.2
                    score.append(current_score.cpu().numpy())
                    loss += 1
                    
                if game_a.verify_result():
                    for k in range(len(next_action_taken)):
                        if score[-(k + 1)][next_action_taken[-(k + 1)]] > 0:
                            score[-(k + 1)][next_action_taken[-(k + 1)]] += ((-1 + DQ_ratio) ** k)
                        else:
                            score[-(k + 1)][next_action_taken[-(k + 1)]] -= ((-1 * DQ_ratio) ** k)                    
                else:
                    torch_tensor *= -1
                    

            input_status_df = pd.DataFrame(input_status)
            input_status_df.to_csv(input_file_path, index=False, mode='a', header=False)
            next_action_taken_df = pd.DataFrame(next_action_taken)
            next_action_taken_df.to_csv(next_action_file_path, index=False, mode='a', header=False)
            score_df = pd.DataFrame(score)
            score_df.to_csv(score_file_path, index=False, mode='a', header=False)

    input_status_df = pd.read_csv(input_file_path, header=None, error_bad_lines=False, warn_bad_lines=True)
    score_df = pd.read_csv(score_file_path, header=None)
    consul_df = pd.concat([input_status_df, next_action_taken_df, score_df], axis=1)
    consul_df = consul_df.replace(-0.0, 0.0)
    consul_df.drop_duplicates(keep='first', inplace=True)
    input_status_df = consul_df.iloc[:, 0:9].copy()
    next_action_taken_df = consul_df.iloc[:, 9:10].copy()
    score_df = consul_df.iloc[:, 10:].copy()
    input_status_df.to_csv(input_file_path, index=False, header=False)
    next_action_taken_df.to_csv(next_action_file_path, index=False, header=False)
    score_df.to_csv(score_file_path, index=False, header=False)

    print("win:", win)
    print("Loss:", loss)


if train_model_from_crash == 1:
    if os.path.exists(file_name_model_latest_version):
        os.remove(file_name_model_latest_version)
    if os.path.exists(file_name_model_last_version):
        os.remove(file_name_model_last_version)
    if os.path.exists(input_file_path):
        os.remove(input_file_path)
    if os.path.exists(score_file_path):
        os.remove(score_file_path)
    if os.path.exists(next_action_file_path):
        os.remove(next_action_file_path)

if button == 1:
    for loops in range(main_loop_count):
        Training_model()
        RL_model()
else:
    RL_model()
