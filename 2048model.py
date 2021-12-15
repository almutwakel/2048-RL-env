import random
from collections import deque
import numpy as np
import gym
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

gamma = 0.9
epsilon = 0.3

"""
1. Create a gym environment
"""
env = gym.make('CartPole-v0')

"""
2. Create a neural network
"""
model = keras.Sequential()
model.add(layers.Dense(24, input_dim=4, activation='relu'))
model.add(layers.Dense(24, activation='relu'))
model.add(layers.Dense(2, activation='relu'))
model.compile(loss='mse', optimizer=keras.optimizers.Adam(learning_rate=0.001))
model.summary()
"""
3. Create a memory
"""
memory = deque(maxlen=2000)

"""
4. Create a function to take random action
"""
def take_random_action():
    return env.action_space.sample()

"""
5. Create a function to remember an experience
"""
def remember_experience(state, action, reward, next_state, done):
    memory.append((state, action, reward, next_state, done))

"""
6. Create a function to replay past experiences
"""
def replay_memory(batch_size):
    if len(memory) < batch_size:
        return
    # select a random batch
    samples = random.sample(memory, batch_size)
    for sample in samples:
        state, action, reward, next_state, done = sample
        target = reward
        if not done:
            target = reward + gamma * np.amax(model.predict(next_state)[0])
        target_f = model.predict(state)
        target_f[0][action] = target
        print(state)
        model.fit(state, target_f, epochs=1, verbose=0)

"""
7. Create a function to play one round
"""
def play_one_round(render=False):
    state = env.reset()
    done = False
    while not done:
        if np.random.rand() < epsilon:
            action = take_random_action()
        else:
            action = np.argmax(model.predict(state))
        next_state, reward, done, _ = env.step(action)
        if render:
            env.render()
        remember_experience(state, action, reward, next_state, done)
        state = next_state

"""
8. Create a function to play many rounds
"""


def play_many_rounds(num_rounds=10000, batch_size=32, gamma=gamma, epsilon=epsilon):
    for i in range(num_rounds):
        play_one_round()
        replay_memory(batch_size)
        if epsilon > 0.01:
            epsilon *= 0.999
        if (i+1) % 100 == 0:
            print('Rounds {} to {} done'.format(max(0, i+1-100), i+1))

"""
9. Create a function to test the model
"""
def test_model(num_rounds=100, render=False):
    total_reward = 0
    for i in range(num_rounds):
        state = env.reset()
        done = False
        while not done:
            action = np.argmax(model.predict(state))
            next_state, reward, done, _ = env.step(action)
            total_reward += reward
            if render:
                env.render()
            state = next_state
    print('Average reward per round = {}'.format(total_reward / num_rounds))

"""
10. Create a function to train and test the model
"""
def train_and_test_model(train_rounds=10000, test_rounds=1000, batch_size=32, gamma=0.9):
    play_many_rounds(train_rounds, batch_size, gamma)
    test_model(test_rounds)

"""
11. Train and test the model
"""
# train_and_test_model()
take_random_action()