import random
from collections import deque
import numpy as np
import gym
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import env as e
import datetime
import os

gamma = 0.9
epsilon = 0.2

"""
1. Create a gym environment
"""
env = e.Env2048()

"""
2. Create a neural network
"""
checkpoint_path = "training_1/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

# Create a callback that saves the model's weights
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)
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
    return [[random.randrange(0, 1), random.randrange(0, 1), random.randrange(0, 1), random.randrange(0, 1)]]

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

def play_one_round(render=True, verbose=False):
    state = env.reset()
    done = False
    while not done:
        if np.random.rand() < epsilon:
            actions = take_random_action()
        else:
            actions = model.predict([state])
        action = np.argmax(actions[0])
        # print("action:", action)
        while not env.play.check_valid_move(action):
            actions[0][action] = -10000
            action = np.argmax(actions[0])
        next_state, reward, done, flags = env.step(action)
        if render:
            env.render()
        if done:
            break
        # print(next_state, reward, done, _)
        # print(action)
        remember_experience([state], action, reward, next_state, done)
        state = next_state



"""
8. Create a function to play many rounds
"""


def play_many_rounds(num_rounds=100, batch_size=32, gamma=gamma, epsilon=epsilon, render=True, verbose=False):
    for i in range(num_rounds):
        timestamp1 = datetime.datetime.now()
        play_one_round(render=render, verbose=verbose)
        timestamp2 = datetime.datetime.now()
        replay_memory(batch_size)
        timestamp3 = datetime.datetime.now()
        if verbose:
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
            action = np.argmax(actions[0])
            # print("action:", action)
            while not env.play.check_valid_move(action):
                actions[0][action] = -10000
                action = np.argmax(actions[0])
            next_state, reward, done, flags = env.step(action)
            total_reward += reward
            if render:
                env.render()
            state = next_state
            if done:
                break
    timestamp2 = datetime.datetime.now()
    print('Average reward per round = {}'.format(total_reward / num_rounds))
    print("Testing time: ", timestamp2-timestamp1)

"""
10. Create a function to train and test the model
"""
def train_and_test_model(train_rounds=10000, test_rounds=1000, batch_size=32, gamma=0.9, render=False):
    print("Training", train_rounds, "rounds:")
    play_many_rounds(train_rounds, batch_size, gamma, epsilon, render=render)
    print("Testing", test_rounds, "rounds:")
    test_model(test_rounds, render=render)

"""
11. Train and test the model
"""
model.summary()
# train_and_test_model()
# test_model(num_rounds=10, render=False)
train_and_test_model(train_rounds=10, test_rounds=10, render=False)
# replay_memory(10)
# play_one_round()
