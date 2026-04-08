import numpy as np
import random


class QAgent:
    def __init__(self):
        self.q_table = np.zeros((11, 11, 2, 2, 2))  # bucketized states + emergency + actions
        self.alpha = 0.1
        self.gamma = 0.9
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.05

    def choose_action(self, state):
        l1, l2, light, emergency = state
        l1 = min(l1 // 5, 10)
        l2 = min(l2 // 5, 10)

        if random.random() < self.epsilon:
            return random.randint(0, 1)
        return int(np.argmax(self.q_table[l1][l2][light][emergency]))

    def learn(self, state, action, reward, next_state):
        l1, l2, light, emergency = state
        nl1, nl2, nlight, nemergency = next_state
        l1 = min(l1 // 5, 10)
        l2 = min(l2 // 5, 10)
        nl1 = min(nl1 // 5, 10)
        nl2 = min(nl2 // 5, 10)

        predict = self.q_table[l1][l2][light][emergency][action]
        target = reward + self.gamma * np.max(self.q_table[nl1][nl2][nlight][nemergency])

        self.q_table[l1][l2][light][emergency][action] += self.alpha * (target - predict)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
