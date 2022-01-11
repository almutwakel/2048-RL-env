import numpy as np
import gym

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Flatten, Conv2D, Permute
from tensorflow.keras.optimizers import Adam

from rl.agents.dqn import DQNAgent
from rl.memory import SequentialMemory
from rl.policy import EpsGreedyQPolicy

from policy import BestValidMovePolicy
import env as e


env = e.Env2048(conv=True)

np.random.seed(123)
# env.seed(123)
nb_actions = env.action_space.n

model = Sequential()
# model.add(Flatten(input_shape=(1,) + env.observation_space.shape))
model.add(Permute((2, 3, 1), input_shape=(1, 4, 4)))
model.add(Conv2D(16, kernel_size=3, activation='relu'))
model.add(Conv2D(16, kernel_size=2, activation='relu'))
model.add(Flatten())
model.add(Dense(4, activation='linear'))
print(model.summary())
print("#osh", model.output.shape)

memory = SequentialMemory(limit=50000, window_length=1)
train_policy = EpsGreedyQPolicy(eps=1)
test_policy = BestValidMovePolicy(env)
dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=100,
               target_model_update=1e-2, policy=train_policy, test_policy=test_policy)
dqn.compile(Adam(learning_rate=1e-3), metrics=['mae'])


dqn.fit(env, nb_steps=10000, visualize=True, verbose=1)

# After training is done, we save the final weights.
dqn.save_weights(f'savedata/dqcnn_2048.h5f', overwrite=True)
# dqn.load_weights(f'savedata/dqn_2048_weights.h5f')

# Finally, evaluate our algorithm for 5 episodes.
print("test")
dqn.test(env, nb_episodes=5, visualize=True)
print("test over")
