import numpy as np
from rl.policy import Policy


class BestValidMovePolicy(Policy):
    """
    Custom Policy for testing on the 2048 environment.
    Uses best valid policy by testing each one until a change is detected.
    """

    def __init__(self, env):
        super().__init__()
        self.env = env

    def select_action(self, q_values):
        """Return the selected action

        # Arguments
            q_values (np.ndarray): List of the estimations of Q for each action

        # Returns
            Selection action
        """
        assert q_values.ndim == 1
        nb_actions = q_values.shape[0]

        if np.random.uniform() < self.eps:
            action = np.random.randint(0, nb_actions)
        else:
            action = np.argmax(q_values)
        return action

