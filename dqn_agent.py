import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque

class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(4, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )

    def forward(self, x):
        return self.net(x)

class DQNAgent:
    def __init__(self):
        self.model = DQN()
        self.target_model = DQN()
        self.target_model.load_state_dict(self.model.state_dict())

        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)

        self.loss_fn = nn.SmoothL1Loss()
        self.grad_clip = 1.0

        self.memory = deque(maxlen=5000)

        self.gamma = 0.9
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.05

        self.batch_size = 32
        self.update_target_every = 50
        self.step_count = 0

    def choose_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0,1)

        state = torch.FloatTensor(state) / 50.0
        q_values = self.model(state)
        return torch.argmax(q_values).item()

    def store(self, state, action, reward, next_state):
        self.memory.append((state, action, reward, next_state))

    def train(self):
        if len(self.memory) < self.batch_size:
            return

        batch = random.sample(self.memory, self.batch_size)

        states, actions, rewards, next_states = zip(*batch)

        states = torch.FloatTensor(states) / 50.0
        next_states = torch.FloatTensor(next_states) / 50.0

        q_values = self.model(states)
        next_q_values = self.target_model(next_states)

        target = q_values.clone().detach()

        for i in range(self.batch_size):
            target[i][actions[i]] = rewards[i] + self.gamma * torch.max(next_q_values[i]).item()

        loss = self.loss_fn(q_values, target)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
        self.optimizer.step()

        # epsilon decay
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        # update target network
        self.step_count += 1
        if self.step_count % self.update_target_every == 0:
            self.target_model.load_state_dict(self.model.state_dict())