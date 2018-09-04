import gym
import numpy as np
from collections import defaultdict

# Make Environment
env = gym.make('Blackjack-v0')

def mc_policy(bj_env):
    episode = []
    state = bj_env.reset()
    player_hand = state[0]
    opp_hand = state[1]
    opp_has_ace = state[2]
    """Policy is, player will stick hand with 80% prob (or hit with 20% prob)
       if sum is greater than 18
       else keep asking for hits with 80% prob
    """
    if player_hand > 18:
       stick_prob = [0.8, 0.2]
    else:
       stick_prob = [0.2, 0.8]
    while True:
        action = np.random.choice(np.arange(2), p=stick_prob)
        next_state, reward, done, info = env.step(action)
        episode.append((next_state, action, reward))
        state = next_state
        if done:
            break
    return episode

def first_visit_mc(bj_env, num_episodes):
    Q = defaultdict
    # Run an episode using mc_policy
    episode = mc_policy(bj_env)
    Q = defaultdict(lambda: np.zeros(bj_env.action_space.n))
    N = defaultdict(lambda: np.zeros(bj_env.action_space.n))
    returns_sum = defaultdict(lambda: np.zeros(bj_env.action_space.n))
    
    return Q

Q = first_visit_mc(env, 10)
print(Q)
                
        
    



