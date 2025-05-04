import torch
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import numpy as np
from models.q_network import QNetwork
from models.replay_buffer import ReplayBuffer

class DQNAgentFinetuned:
    def __init__(self, state_dim, action_dim, gamma=0.90, lr=5e-4, batch_size=128):
        self.q_net = QNetwork(state_dim, action_dim)
        self.target_net = QNetwork(state_dim, action_dim)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.AdamW(self.q_net.parameters(), lr=lr, weight_decay=1e-2)

        self.scheduler = StepLR(self.optimizer, step_size=10, gamma=0.5)

        self.buffer = ReplayBuffer()
        self.gamma = gamma
        self.batch_size = batch_size
        self.loss_fn = torch.nn.MSELoss()

        self.epsilon = 1.0
        self.epsilon_decay = 0.98
        self.epsilon_min = 0.01

        self.loss_history = []

    def select_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.randint(0, 3)
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
        self.loss_history.append(loss.item())

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.scheduler.step()

        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)

    def update_target(self):
        self.target_net.load_state_dict(self.q_net.state_dict())
