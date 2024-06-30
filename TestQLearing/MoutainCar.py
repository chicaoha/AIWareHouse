import gym
import pygame
import numpy as np


env = gym.make("MountainCar-v0")
env.reset()
c_learning_rate = 0.1
c_discount_value = 0.9
c_no_of_eps = 500
c_show_each = 500

q_table_size = [20, 20]
q_table_segment_size = (env.observation_space.high - env.observation_space.low) / q_table_size


# print(q_table_segment_size)


# ham chuyen doi real state ve q_state

def convert_state(real_state):
    q_state = (real_state - env.observation_space.low) // q_table_segment_size
    return tuple(q_state.astype(int))


# khoi tao table
q_table = np.random.uniform(low=-2, high=0, size=(q_table_size + [env.action_space.n]))

max_ep_reward = -999
max_ep_action_list = []
for ep in range(c_no_of_eps):
    print("Eps = :", ep)
    done = False
    # Reset environment and get initial state
    initial_state = env.reset()
    state = initial_state if isinstance(initial_state, np.ndarray) else initial_state[0] if isinstance(initial_state,
                                                                                                       tuple) else \
        initial_state['observation']

    current_state = convert_state(state)
    ep_reward = 0
    action_list =[]
    if ep % c_show_each == 0:
        show_now = True
    else:
        show_now = False
    while not done:
        # lay max Q value cua current state
        action = np.argmax(q_table[current_state])
        action_list.append(action)
        # hanh dong the action da lay
        result = env.step(action)
        # next_real_state = result[0]
        # reward = result[1]
        # done = result[2]
        # info = result[3]
        # Handle different return types of step function

        if len(result) == 4:
            next_real_state, reward, done, info = result
            ep_reward += reward
        else:
            next_real_state, reward, done, truncated, info = result
            done = done or truncated
        if show_now:
            env.render()
        if done:
            # kiem tra vi tri x co lon hon la co khong
            if next_real_state[0] >= env.goal_position:

                print("Da den co tai ep ={}, reward ={}".format(ep, reward))
                if(ep_reward > max_ep_reward):
                    max_ep_reward = ep_reward
                    max_ep_action_list = action_list
        else:
            # convert ve  q_state
            next_state = convert_state(next_real_state)

            # update Q value cho current state and action
            current_q_value = q_table[current_state + (action,)]
            new_q_value = (1 - c_learning_rate) * current_q_value + c_learning_rate * (
                    reward + c_discount_value * np.max(q_table[next_state]))
            q_table[current_state + (action,)] = new_q_value
            current_state = next_state

print("Max reward = ", max_ep_reward)
print("Max action List", max_ep_action_list)