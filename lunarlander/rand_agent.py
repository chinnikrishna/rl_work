import numpy as np

class RandomAgent:
    """
    Random Agent which generates valid random actions
    """
    def __init__(self, state_space, action_space):
        self.state_space = state_space
        self.action_space = action_space

    def get_action(self):
        # Generates a valid random action
        action = np.random.randint(low=0, high=self.action_space)
        return action
