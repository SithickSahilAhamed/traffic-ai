import random
from models import Action, Observation

# Task-specific configuration
TASK_CONFIG = {
    "easy":   {"max_steps": 20, "emergency_prob": 0.00, "arrival_rate": 2},
    "medium": {"max_steps": 30, "emergency_prob": 0.10, "arrival_rate": 4},
    "hard":   {"max_steps": 40, "emergency_prob": 0.25, "arrival_rate": 7},
}


class TrafficEnv:
    def __init__(self):
        self.max_cars = 50
        self._config = TASK_CONFIG["easy"]
        self.reset()

    # BUG FIX 1: accept task_id so inference.py / app.py don't crash
    def reset(self, task_id: str = "easy"):
        self._config = TASK_CONFIG.get(task_id, TASK_CONFIG["easy"])
        self.task_id = task_id

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

        # BUG FIX 2: action semantics corrected
        #   0 = give green to lane1  (was: 0=stay)
        #   1 = give green to lane2  (was: 1=switch)
        self.light = action_value

        # BUG FIX 3: emergency gives priority to lane2 (action=1), not lane1
        if self.emergency == 1:
            self.light = 1  # was: self.light = 0 (forced lane1, which is wrong)

        # Cars passing through green lane
        if self.light == 0:
            passed = min(self.lane1, 3)
            self.lane1 -= passed
        else:
            passed = min(self.lane2, 3)
            self.lane2 -= passed

        # Incoming traffic based on task difficulty
        arrival = self._config["arrival_rate"]
        self.lane1 = min(self.lane1 + random.randint(0, arrival), self.max_cars)
        self.lane2 = min(self.lane2 + random.randint(0, arrival), self.max_cars)

        # Emergency event for next step
        self.emergency = 1 if random.random() < self._config["emergency_prob"] else 0

        # Reward: penalize congestion
        reward = -(self.lane1 + self.lane2)

        # Bonus/penalty for emergency handling
        if self.emergency == 1:
            if self.light == 1:   # correct: gave green to lane2
                reward += 10
            else:                 # wrong: ignored emergency
                reward -= 10

        self.time += 1

        # BUG FIX 4: done depends on task's max_steps, not hardcoded 100
        done = self.time >= self._config["max_steps"]

        return self.state(), reward, done, {"step": self.time, "task_id": self.task_id}
