import numpy as np
from tqdm import tqdm


class Walk2D:
    def __init__(self, maxx=10, maxy=10):
        self.state = None
        self.maxx = maxx
        self.maxy = maxy

    def reset(self, startx=0, starty=0):
        self.state = (startx, starty)
        reward = 0.0
        done = False
        return self.state, reward, done

    def step(self, action):
        go_left, go_right, go_up, go_down = 0, 1, 2, 3
        done = False
        reward = 0.0
        if action == go_left:
            state = (self.state[0] - 1, self.state[1])
        elif action == go_right:
            state = (self.state[0] + 1, self.state[1])
        elif action == go_up:
            state = (self.state[0], self.state[1] + 1)
        elif action == go_down:
            state = (self.state[0], self.state[1] - 1)
        else:
            state = None
            print(state, action)
            raise ValueError("This should not happen")

        if (self.state[0] == 0 and action == go_left) or \
           (self.state[1] == 0 and action == go_down) or \
           (self.state[0] == self.maxx and action == go_right) or \
           (self.state[1] == self.maxy and action == go_up):
            reward = -1.0
            done = False
        elif (((self.state[0] + 1, self.state[1]) == (self.maxx, self.maxy)) and action == go_right) or \
           (((self.state[0], self.state[1] + 1) == (self.maxx, self.maxy)) and action == go_up):
            self.state = (self.maxx, self.maxy)
            done = True
            reward = self.maxx * self.maxy * 100
        else:
            self.state = state
            reward = -1.0
            done = False
        return self.state, reward, done

np.random.seed(0)
maxx, maxy = 2, 2
env = Walk2D(maxx=maxx, maxy=maxy)
# Parameters
eps = 0.1
gamma = 0.9
num_episodes = 10

max_timestamp = 100
avf = np.zeros(shape=(maxx, maxy, 4))
print(avf)

for episode in range(num_episodes):
    print("Episode " + str(episode))
    state, reward, done = env.reset(startx=0, starty=0)
    timestamp = 0
    while not done:
        if np.random.random() > eps:
            action = np.random.choice(4)
        else:
            action = np.argmax(avf[state[0]][state[1]])
        next_state, reward, done = env.step(action)
        td_target = reward + gamma * avf[next_state[0]][next_state[1]] * (not done)
        td_error = td_target - avf[state[0]][state[1]]
        avf[state[0]][state[1]] += td_error
        state = next_state
        timestamp += 1
        if timestamp > max_timestamp:
            print("Broke")
            break
        if done:
            print("Done")
    print(avf)
import pdb
pdb.set_trace()


