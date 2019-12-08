"""
Code for instantiating different agents and running them
"""
import gym
import torch
import matplotlib.pyplot as plt
import numpy as np
from collections import deque
from rand_agent import RandomAgent
from dqn_agent import DQNAgent


def run_rand_agent(env, agent):
    state = env.reset()
    while True:
        action = agent.get_action()
        env.render()
        state, reward, done, misc = env.step(action)
        if done:
            break
    env.close()

def run_dqn_agent(env, agent, num_episodes=2000, max_t=1000):
    scores = []
    scores_window = deque(maxlen=1000)
    eps = 1.0 # Pro exploration
    for i_episode in range(1, num_episodes + 1):
        state = env.reset()
        score = 0
        for ts in range(max_t):
            # Get action from agent
            action = agent.get_action(state, eps)
            # Render
            # env.render()
            # Run through sim
            next_state, reward, done, _ = env.step(action)
            # Process the observations
            agent.process_obs(state, action, reward, next_state, done)
            state = next_state
            score += reward
            if done:
                break
        scores_window.append(score)
        scores.append(score)
        # Decrease epsilon
        eps = max(0.01, 0.995*eps)
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)), end="")
        if i_episode % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
        if np.mean(scores_window)>=200.0:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode-100, np.mean(scores_window)))
            torch.save(agent.qnet_policy.state_dict(), 'checkpoint.pth')
            break
    return scores

env = gym.make('LunarLander-v2')
env = gym.wrappers.Monitor(env, './video/', force = True)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
seed = 0
agent = DQNAgent(device, env.observation_space.shape[0], env.action_space.n, seed)
agent.qnet_policy.load_state_dict(torch.load('./checkpoint.pth'))
for i in range(10):
    state = env.reset()
    for j in range(1500):
        action = agent.get_action(state)
        #env.render()
        state, reward, done, _ = env.step(action)
        if done:
            break
#scores = run_dqn_agent(env, agent)

# plot the scores
# fig = plt.figure()
# ax = fig.add_subplot(111)
# plt.plot(np.arange(len(scores)), scores)
# plt.ylabel('Score')
# plt.xlabel('Episode #')
# plt.show()

