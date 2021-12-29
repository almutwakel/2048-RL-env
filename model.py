import random
from collections import deque
import numpy as np
import gym
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import env as e
import datetime

gamma = 0.9
epsilon = 0.2

"""
1. Create a gym environment
"""
env = e.Env2048()

"""
2. Create a neural network
"""
model = keras.Sequential()
model.add(layers.Dense(16, input_dim=16, activation='relu'))
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(4, activation='relu'))
model.compile(loss='mse', optimizer=keras.optimizers.Adam(learning_rate=0.001))
# model.summary()
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
        # print("next state:", next_state)
        # next_state = np.reshape(next_state, (16, ))
        # print(next_state.ndim)
        if not done:
            target = reward + gamma * np.amax(model.predict([next_state], batch_size=1)[0])
        target_f = model.predict(state)
        target_f[0][action] = target
        # print("targetf:", target_f)
        # print("state:", state)
        model.fit(np.array(state), np.array(target_f), epochs=1, verbose=0)

"""
7. Create a function to play one round
"""

def play_one_round(render=True):
    state = env.reset()
    done = False
    while not done:
        timestamp1 = datetime.datetime.now()
        if np.random.rand() < epsilon:
            action = take_random_action()
        else:
            # action = take_random_action()
            action = np.argmax(model.predict([state]))
        timestamp2 = datetime.datetime.now()
        next_state, reward, done, _ = env.step(action)
        timestamp3 = datetime.datetime.now()
        # print(next_state, reward, done, _)
        if render:
            env.render()
        timestamp4 = datetime.datetime.now()
        remember_experience([state], action, reward, next_state, done)
        state = next_state
        timestamp5 = datetime.datetime.now()
        print("Action time:", timestamp2 - timestamp1)
        print("Step time:", timestamp3 - timestamp2)
        print("Render time:", timestamp4 - timestamp3)
        print("Remember time:", timestamp5 - timestamp4)



"""
8. Create a function to play many rounds
"""


def play_many_rounds(num_rounds=100, batch_size=32, gamma=gamma, epsilon=epsilon):
    for i in range(num_rounds):
        timestamp1 = datetime.datetime.now()
        play_one_round()
        timestamp2 = datetime.datetime.now()
        replay_memory(batch_size)
        timestamp3 = datetime.datetime.now()
        print("> Total round play time:", timestamp2 - timestamp1)
        print("> Round model time:", timestamp3 - timestamp2)
        if epsilon > 0.01:
            epsilon *= 0.999
        if (i+1) % 100 == 0:
            print('Rounds {} to {} done'.format(max(0, i+1-100), i+1))

"""
9. Create a function to test the model
"""
def test_model(num_rounds=100, render=False):
    timestamp1 = datetime.datetime.now()
    total_reward = 0
    for i in range(num_rounds):
        state = env.reset()
        done = False
        while not done:
            actions = model.predict([state])
            changed = False
            while not changed:
                action = np.argmax(actions)
                print("action picked:", action, "from", actions)
                next_state, reward, done, flags = env.step(action)
                changed = flags["changed"]
                actions[0][action] = -1
                print("stepped:", reward, "changed", changed)
                total_reward += reward
                if render:
                    env.render()
                    print("rendered")
                state = next_state
                if done:
                    break
    timestamp2 = datetime.datetime.now()
    print('Average reward per round = {}'.format(total_reward / num_rounds))
    print("Testing time: ", timestamp2-timestamp1)

"""
10. Create a function to train and test the model
"""
def train_and_test_model(train_rounds=10000, test_rounds=1000, batch_size=32, gamma=0.9):
    play_many_rounds(train_rounds, batch_size, gamma)
    test_model(test_rounds)

"""
11. Train and test the model
"""
model.summary()
# train_and_test_model()
test_model(num_rounds=10, render=False)
# train_and_test_model(train_rounds=100, test_rounds=5)
# replay_memory(10)
# play_one_round()
# play_one_round()
