# Imports
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


class Walk1D:
    def __init__(self, max_state):
        self.state = None
        self.max_state = max_state
        self.go_left = 0
        self.go_right = 1

    def reset(self, start_state=0):
        self.state = start_state
        reward = [self.state if self.state == self.max_state else 0]
        done = [True if self.state == self.max_state else False]
        return self.state, reward[0], done[0]

    def step(self, action):
        done = False
        reward = 0.0
        if (action == self.go_left and self.state != 0):
            self.state -= 1
            done = False
            reward = -1.0
        elif (action == self.go_left and self.state == 0):
            done = False
            reward = -1.0
        elif (action == self.go_right and (self.state + 1) != self.max_state):
            self.state += 1
            done = False
            reward = -1.0
        elif (action == self.go_right and (self.state + 1) == self.max_state):
            self.state = self.max_state
            done = True
            reward = self.max_state * 100
        else:
            print(action, self.state, done)
            raise ValueError("This should not happen")
        return self.state, reward, done


env = Walk1D(max_state=2000)
# Parameters
eps = 0.1
gamma = 0.9
num_episodes = 100
max_timestamp = 10000
avf = np.zeros(shape=(env.max_state + 1, 2))
num_steps = []
for episode in range(num_episodes):
    print("Episode: " + str(episode))
    state, reward, done = env.reset()
    timestamp = 0
    while not done:
        if np.random.random() > eps:
            action = np.random.choice(2)
        else:
            action = np.argmax(avf[state])
        next_state, reward, done = env.step(action)
        td_target = reward + gamma * avf[next_state][action] * (not done)
        td_error = td_target - avf[state][action]
        avf[state][action] = avf[state][action] +  td_error
        state = next_state
        timestamp += 1
        if timestamp > max_timestamp:
            print("Break")
            break
        if done:
            print("Done")
    num_steps.append(timestamp)
for state in avf:
    print(np.argmax(state), end=" ")
# plt.plot([x for x in range(num_episodes)], num_steps)
# plt.show()

