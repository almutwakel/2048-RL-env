import numpy as np
import gym

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Flatten, Conv2D
from tensorflow.keras.optimizers import Adam

from rl.agents.dqn import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory

import env as e


# Get the environment and extract the number of actions.
env = e.Env2048(conv=True)

np.random.seed(123)
# env.seed(123)
nb_actions = env.action_space.n

# Next, we build a very simple model.
model = Sequential()
# model.add(Flatten(input_shape=(1,) + env.observation_space.shape))
model.add(Conv2D(16, kernel_size=3, activation='relu', input_shape=(4, 4, 1)))
model.add(Conv2D(16, kernel_size=2, activation='relu'))
model.add(Flatten())
model.add(Dense(nb_actions, activation='softmax'))
print(model.summary())
print("#osh", model.output.shape)
# Finally, we configure and compile our agent. You can use every built-in tensorflow.keras optimizer and
# even the metrics!
memory = SequentialMemory(limit=50000, window_length=1)
policy = BoltzmannQPolicy()
dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=100,
               target_model_update=1e-2, policy=policy)
dqn.compile(Adam(learning_rate=1e-3), metrics=['mae'])

# Okay, now it's time to learn something! We visualize the training here for show, but this
# slows down training quite a lot. You can always safely abort the training prematurely using
# Ctrl + C.
dqn.fit(env, nb_steps=10000, visualize=True, verbose=1)

# After training is done, we save the final weights.
dqn.save_weights(f'savedata/dqn_2048_weights.h5f', overwrite=True)

# Finally, evaluate our algorithm for 5 episodes.
print("test")
dqn.test(env, nb_episodes=1, visualize=True)
print("test over")
