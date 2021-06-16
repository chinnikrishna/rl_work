import numpy as np


class BSW:
    def __init__(self):
        self.bsw_env = {
            0: {
                0:[(1.0, 0, 0.0, True)],
                1:[(1.0, 0, 0.0, True)]
            },
            1: {
                0:[(0.50, 0, 0.0, False),
                   (0.34, 1, 0.0, False),
                   (0.16, 2, 0.0, False)],
                1:[(0.50, 2, 0.0, False),
                   (0.34, 1, 0.0, False),
                   (0.16, 0, 0.0, False)],
             },
            2: {
                0:[(0.50, 1, 0.0, False),
                   (0.34, 2, 0.0, False),
                   (0.16, 3, 0.0, False)],
                1:[(0.50, 3, 0.0, False),
                   (0.34, 2, 0.0, False),
                   (0.16, 1, 0.0, False)],
             },
            3: {
                0:[(0.50, 2, 0.0, False),
                   (0.34, 3, 0.0, False),
                   (0.16, 4, 0.0, False)],
                1:[(0.50, 4, 0.0, False),
                   (0.34, 3, 0.0, False),
                   (0.16, 2, 0.0, False)],
             },
            4: {
                0:[(0.50, 3, 0.0, False),
                   (0.34, 4, 0.0, False),
                   (0.16, 5, 0.0, False)],
                1:[(0.50, 5, 0.0, False),
                   (0.34, 4, 0.0, False),
                   (0.16, 3, 0.0, False)],
             },
            5: {
                0:[(0.50, 4, 0.0, False),
                   (0.34, 5, 0.0, False),
                   (0.16, 6, 1.0, True)],
                1:[(0.50, 6, 1.0, True),
                   (0.34, 5, 0.0, False),
                   (0.16, 4, 0.0, False)],
             },
            6: {
                0:[(1.0, 6, 0.0, True)],
                1:[(1.0, 6, 0.0, True)]
            }
        }
        self.state = None
        self.action = None
        
    def reset(self, start_state=3):
        self.state = start_state
        reward = [1.0 if self.state == 6 else 0.0]
        done = [True if (self.state == 6 or self.state==0) else False]
        return self.state, reward[0], done[0]
        
    def step(self, action):
        # Get tuple of next possible states
        transition_tuples = self.bsw_env[self.state][action]
        # Separate tuple into prob, next states, rewards and done
        trans_probs = [i[0] for i in transition_tuples]
        nps = [i[1] for i in transition_tuples]
        rewards = [i[2] for i in transition_tuples]
        done_vals = [i[3] for i in transition_tuples]
        # Select next state index based on trans_probs
        ns_idx = np.random.choice([i for i in range(len(transition_tuples))], p=trans_probs)
        # Set state of environment
        self.state = nps[ns_idx]
        # Storing action for render
        self.action = action
        return (self.state, rewards[ns_idx], done_vals[ns_idx])
    
    def render(self, print_state_names=False):
        if print_state_names:
            print("|  H-0  |  1  |  2  |  3  |  4  |  5  |  G-6  |")
        for i in range(7):
            str_to_print = "|  "
            if i == 0:
                str_to_print += "  "
            if i == self.state:
                str_to_print += "A"
                str_to_print += "  "
            else:
                str_to_print += "   "
            if i == 6:
                str_to_print += "  |"
                if self.action is not None:
                    if self.action == 0:
                        str_to_print += " Left"
                    if self.action == 1:
                        str_to_print += " Right"
            print(str_to_print, end='')
        print("")
