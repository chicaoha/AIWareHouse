import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import random
import os

# Frame dimensions
ROWS = 7
COLS = 14

# Item types with their respective weights
ITEM_TYPES = {0: 'white', -1: 'green', 1: 'red', 2: 'beige'}
ITEM_WEIGHTS = {0: 0, -1: 0.5, 1: 1, 2: 1.5}

# Target ratio
TARGET_RATIO = {0: (2/7), 1: (1/7), 2: (2/7)}

# Hyperparameters
GAMMA = 0.99
LEARNING_RATE = 0.001
EPISODES = 3
EPSILON = 1.0
EPSILON_MIN = 0.01
EPSILON_DECAY = 0.995
BATCH_SIZE = 32
MEMORY_SIZE = 2000
MODEL_PATH = 'dqn_model.h5'

# Score table
SCORE_TABLE = np.array([
    [14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1],
    [14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1],
    [14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1],
    [14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1],
    [14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1],
    [14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1],
    [14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1]
])
# Weight table
WEIGHT_TABLE = np.array([
    [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
    [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]
])
# Create the Deep Q-Network
def build_model():
    model = Sequential()
    model.add(Dense(24, input_dim=ROWS*COLS, activation='relu'))
    model.add(Dense(24, activation='relu'))
    model.add(Dense(len(ITEM_TYPES), activation='linear'))
    model.compile(loss='mse', optimizer=Adam(learning_rate=LEARNING_RATE))
    return model

class DQNAgent:
    def __init__(self):
        self.model = build_model()
        self.target_model = build_model()
        self.memory = []
        self.epsilon = EPSILON

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        if len(self.memory) > MEMORY_SIZE:
            self.memory.pop(0)

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.choice(list(ITEM_TYPES.keys()))
        q_values = self.model.predict(state, verbose=0)
        return np.argmax(q_values[0])

    def train(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target += GAMMA * np.amax(self.target_model.predict(next_state, verbose=0)[0])
            target_f = self.model.predict(state, verbose=0)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > EPSILON_MIN:
            self.epsilon *= EPSILON_DECAY

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

def get_initial_frame():
    return np.zeros((ROWS, COLS), dtype=int)

def is_valid_position(frame, row, col):
    return 0 <= row < ROWS and 0 <= col < COLS and frame[row][col] == 0

def place_item_weighted(frame, item, score_table, weight_table):
    if item not in ITEM_WEIGHTS:
        return False

    weight = ITEM_WEIGHTS[item]
    valid_positions = []

    for row in range(ROWS):
        for col in range(COLS):
            if frame[row][col] == 0 and weight_table[row][col] == weight:
                valid_positions.append((row, col, score_table[row][col]))

    if not valid_positions:
        return False

    row, col, _ = max(valid_positions, key=lambda x: x[2])
    frame[row][col] = item
    return True

def calculate_reward(frame, score_table):
    item_counts = {item: np.count_nonzero(frame == item) for item in ITEM_TYPES}
    total_items = sum(item_counts.values())
    if total_items == 0:
        return 0
    
    reward = 0
    for item, count in item_counts.items():
        actual_ratio = count / total_items
        target_ratio = TARGET_RATIO.get(item, 0)
        reward += 1 - abs(actual_ratio - target_ratio)
    
    total_score = sum(score_table[frame == item].sum() for item in ITEM_TYPES)
    reward += total_score / (ROWS * COLS * 14)  # Normalize by the max possible score (14)
    
    return reward / len(ITEM_TYPES)

def reshape_frame(frame):
    return np.reshape(frame, [1, ROWS * COLS])

def train_dqn():
    agent = DQNAgent()
    for episode in range(EPISODES):
        frame = get_initial_frame()
        state = reshape_frame(frame)
        total_reward = 0
        for step in range(ROWS * COLS):
            action = agent.act(state)
            if place_item_weighted(frame, action,SCORE_TABLE, WEIGHT_TABLE):
                next_state = reshape_frame(frame)
                reward = calculate_reward(frame,SCORE_TABLE)
                done = step == ROWS * COLS - 1
                agent.remember(state, action, reward, next_state, done)
                state = next_state
                total_reward += reward
                if done:
                    break
            if len(agent.memory) > BATCH_SIZE:
                agent.train(BATCH_SIZE)
        agent.update_target_model()
        print(f"Episode {episode+1}/{EPISODES}, Total Reward: {total_reward}, Epsilon: {agent.epsilon:.2f},state,{state}")
    
    # Save the trained model
    agent.model.save(MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")

if __name__ == "__main__":
    train_dqn()
