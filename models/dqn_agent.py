import torch
import torch.optim as optim
import numpy as np
import os
from models.q_network import QNetwork
from models.q_network_2 import QNetwork as QNetwork2
from models.replay_buffer import ReplayBuffer

class DQNAgent:
    def __init__(self, state_dim, action_dim, gamma=0.99, lr=1e-3, batch_size=64):
        self.q_net = QNetwork(state_dim, action_dim)
        self.target_net = QNetwork(state_dim, action_dim)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.buffer = ReplayBuffer()
        self.gamma = gamma
        self.batch_size = batch_size
        self.loss_fn = torch.nn.MSELoss()
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01

    def select_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.randint(0, 7)
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            return torch.argmax(self.q_net(state_tensor)).item()

    def train(self):
        if len(self.buffer) < self.batch_size:
            return

        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)

        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones)

        q_values = self.q_net(states)
        next_q_values = self.target_net(next_states)

        q_target = q_values.clone()
        for i in range(self.batch_size):
            q_target[i, actions[i]] = rewards[i] + self.gamma * torch.max(next_q_values[i]) * (1 - dones[i])

        loss = self.loss_fn(q_values, q_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)

    def update_target(self):
        self.target_net.load_state_dict(self.q_net.state_dict())
        
    def save_model(self, path):
        """
        Save the model state dictionary to the given path.
        
        Args:
            path (str): Path where the model will be saved
        """
        # Create the directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Save the model state
        save_dict = {
            'q_net_state_dict': self.q_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon
        }
        torch.save(save_dict, path)
        
    def load_model(self, path):
        """
        Load the model state dictionary from the given path.
        
        Args:
            path (str): Path where the model is saved
        """
        if not os.path.exists(path):
            print(f"Model file {path} does not exist.")
            return False
            
        # Load the model state
        checkpoint = torch.load(path)
        self.q_net.load_state_dict(checkpoint['q_net_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        return True
