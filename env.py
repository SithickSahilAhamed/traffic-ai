import random

from models import Action, Observation


class TrafficEnv:
    def __init__(self):
        self.max_cars = 50
        self.reset()

    def reset(self):
        self.lane1 = 10
        self.lane2 = 10
        self.light = 0
        self.emergency = 0
        self.time = 0
        return self.state()

    def state(self):
        return Observation(
            lane1=self.lane1,
            lane2=self.lane2,
            light=self.light,
            emergency=self.emergency,
        )

    def _get_state(self):
        return self.state()

    def step(self, action):
        action_value = action.action if isinstance(action, Action) else int(action)

        # action: 0 = stay, 1 = switch
        if action_value == 1:
            self.light = 1 - self.light

        # emergency override (after action)
        if self.emergency == 1:
            self.light = 0

        # cars passing
        if self.light == 0:
            passed = min(self.lane1, 3)
            self.lane1 -= passed
        else:
            passed = min(self.lane2, 3)
            self.lane2 -= passed

        # new incoming traffic (rush simulation)
        if self.time < 50:
            self.lane1 += random.randint(0, 2)
            self.lane2 += random.randint(0, 2)
        else:
            self.lane1 += random.randint(0, 2)
            self.lane2 += random.randint(0, 2)

        self.lane1 = min(self.lane1, self.max_cars)
        self.lane2 = min(self.lane2, self.max_cars)

        switch_penalty = -5 if action_value == 1 else 0

        # reward (critical)
        reward = -(self.lane1 + self.lane2) + switch_penalty

        if self.emergency == 1:
            if self.light == 0:
                reward += 20
            else:
                reward -= 20

        self.time += 1
        done = self.time >= 100

        return self.state(), reward, done, {}
