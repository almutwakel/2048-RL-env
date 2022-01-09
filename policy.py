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
        original_q = np.copy(q_values)
        for _ in range(nb_actions):
            action = np.argmax(q_values)
            if self.env.play.check_valid_move(action):
                return action
            else:
                q_values[action] = -10000
        # print("CGO:", self.env.play.check_game_over())
        print(Exception, "All moves invalid")
        print(original_q, "->", q_values)
