"""
Deep Q Network based Agent for solving lunar lander gym environment
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
from collections import namedtuple, deque


class DQNAgent:
    def __init__(self, device, state_size, action_size, seed, batch_size=64, lr=5e-4,
                 update_step=4, gamma=0.99, tau=1e-3):
        super(DQNAgent, self).__init__()
        self.device = device
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)
        self.batch_size = batch_size
        self.update_step = update_step
        self.gamma = gamma
        self.tau = tau

        # Network to act as policy
        self.qnet_policy = QNet(state_size, action_size, seed, batch_size).to(device)
        # Taget network
        self.qnet_target = QNet(state_size, action_size, seed, batch_size).to(device)
        # Replay buffer to store experiences
        self.replay_buffer = ReplayBuffer(device, seed)

        # Num steps
        self.time_step = 0

        # Adam Optimizer
        self.optimizer = optim.Adam(self.qnet_policy.parameters(), lr=lr)

    def get_action(self, state, eps=0.0):
        """
        Start with a random policy, collect samples into replay buffer and learn from them.
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)

        # Get an action from policy network
        self.qnet_policy.eval()
        with torch.no_grad():
            action = self.qnet_policy(state)
        
        # Epsilon Greedy action selection for explore or exploit
        if random.random() > eps: # Greedily choose an action
            return np.argmax(action.cpu().data.numpy())
        else: # Take a random action to explore
            return random.choice(np.arange(self.action_size))

    def process_obs(self, state, action, reward, next_state, done):
        # Store the SARS'D in replay buffer
        self.replay_buffer.add(state, action, reward, next_state, done)

        # Check if it has been update_step number of steps
        self.time_step += 1
        if self.time_step % self.update_step == 0:
            # Check if replay buffer has more than batch size number of samples
            if len(self.replay_buffer) > self.batch_size:
                # Randomly sample batch_size number of samples
                experiences = self.replay_buffer.sample_batch()
                # Use these samples to update policy network
                self.learn_params(experiences)
    
    def learn_params(self, experiences):
        """
        Update policy network using the following SARSA Max rule
        $Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \alpha((R_{t+1} + \gamma Q(S_{t+1}, A_{t+1})) - Q(S_t, A_t))
        """
        # Unpack data
        states, actions, rewards, next_states, dones = experiences

        # Get Max Q value for the S_{t+1} by using target network
        Q_targets_next = self.qnet_target(next_states).detach().max(1)[0].unsqueeze(1)
        # Get Q values for current states using bellman equation
        Q_targets = rewards + (self.gamma * Q_targets_next * (1 - dones))
        # Get expected Q values from policy network. What we expect them to be
        Q_expected = self.qnet_policy(states).gather(1, actions)

        # Compute loss and minimize it
        loss = F.mse_loss(Q_expected, Q_targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update the target network with soft update to quell noise 
        #         θ_target = τ*θ_local + (1 - τ)*θ_target
        for target_param, local_param in zip(self.qnet_target.parameters(), self.qnet_policy.parameters()):
            target_param.data.copy_(self.tau*local_param.data + (1.0-self.tau)*target_param.data)

 
class QNet(nn.Module):
    """
    QNet approxiamates the policy
    QNet starts with random weights thus acts as a random policy
    As we collect more samples it learns and gets better via TD Learning
    """
    def __init__(self, state_size, action_size, seed, batch_size=64):
        super(QNet, self).__init__()
        self.seed = torch.manual_seed(seed)
        # MLP to act as a Q-network
        self.net = nn.Sequential(
            nn.Linear(state_size, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, action_size),
        )
    
    def forward(self, state):
        action = self.net(state)
        return action



class ReplayBuffer:
    def __init__(self, device, seed, buffer_size=int(1e5), batch_size=64):
        super(ReplayBuffer, self).__init__()
        self.seed = random.seed(seed)
        self.batch_size = batch_size
        self.device = device
        
        # Buffer modeled as a deque
        self.buffer = deque(maxlen=buffer_size)

        # Named tuple for easier retrieval later
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])

    def add(self, state, action, reward, next_state, done):
        e = self.experience(state, action, reward, next_state, done)
        self.buffer.append(e)
    
    def sample_batch(self):
        """
        Randomly sample batch size of experiences from memory
        """
        experiences = random.sample(self.buffer, k=self.batch_size)
        
        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(self.device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(self.device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(self.device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(self.device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None])).float().to(self.device)

        return (states, actions, rewards, next_states, dones)
    
    def __len__(self):
        return len(self.buffer)
        