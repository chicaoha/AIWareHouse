import gym
import pygame
import numpy as np
np.bool = np.bool_
env = gym.make("MountainCar-v0", render_mode = "human")
env.reset()

print(env.state)

print(env.action_space.n)

#lay x toi thieu x toi da va van toc toi thieu dtoi da

print(env.observation_space.high)
print(env.observation_space.low)

while True:
    action = 2 # thu luon di ve ben phai
    result = env.step(action)
    print("New state = {}".format(result))
    env.render()


