import torch
import torch.optim as optim
import numpy as np
import os
from models.dueling_q_network import DuelingQNetwork
from models.replay_buffer import ReplayBuffer
from models.dqn_agent import DQNAgent

class DuelingDQNAgent(DQNAgent):
    """
    Dueling DQN Agent that extends the base DQNAgent class.
    """
    def __init__(self, state_dim, action_dim, gamma=0.99, lr=1e-3, batch_size=64):
        """
        Initialize the Dueling DQN Agent.
        
        Args:
            state_dim (int): Dimension of the state space
            action_dim (int): Dimension of the action space
            gamma (float): Discount factor
            lr (float): Learning rate
            batch_size (int): Batch size for training
        """
        # Don't call the parent constructor to avoid creating regular Q-networks
        # super().__init__(state_dim, action_dim, gamma, lr, batch_size)
        
        # Create dueling networks
        self.q_net = DuelingQNetwork(state_dim, action_dim)
        self.target_net = DuelingQNetwork(state_dim, action_dim)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

        # Initialize optimizer, replay buffer, and other parameters
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.buffer = ReplayBuffer()
        self.gamma = gamma
        self.batch_size = batch_size
        self.loss_fn = torch.nn.MSELoss()
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        
    # All other methods are inherited from DQNAgent
    # select_action, train, update_target, save_model, load_model 