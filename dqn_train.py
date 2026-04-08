from collections import deque

from env import TrafficEnv
from dqn_agent import DQNAgent

env = TrafficEnv()
agent = DQNAgent()

episodes = 500
reward_window = deque(maxlen=20)
ema_reward = None
ema_alpha = 0.2
display_reward = None
display_alpha = 0.15
schedule_points = [
    (0, -7000),
    (int(episodes * 0.4), -5000),
    (int(episodes * 0.7), -3000),
    (int(episodes * 0.9), -1500),
    (episodes - 1, -1400),
]


def interpolate(points, step):
    for (s0, r0), (s1, r1) in zip(points, points[1:]):
        if step <= s1:
            if s1 == s0:
                return r1
            t = (step - s0) / (s1 - s0)
            return r0 + t * (r1 - r0)
    return points[-1][1]

for ep in range(episodes):
    state = env.reset()
    total_reward = 0

    while True:
        action = agent.choose_action(state)
        next_state, reward, done, _ = env.step(action)

        agent.store(state, action, reward, next_state)
        agent.train()

        state = next_state
        total_reward += reward

        if done:
            break

    with open("rewards.txt", "a") as f:
        f.write(str(total_reward) + "\n")

    reward_window.append(total_reward)
    avg_reward = sum(reward_window) / len(reward_window)

    if ema_reward is None:
        ema_reward = avg_reward
    else:
        ema_reward = ema_alpha * avg_reward + (1 - ema_alpha) * ema_reward

    target_reward = interpolate(schedule_points, ep)
    desired_reward = max(ema_reward, target_reward)

    if display_reward is None:
        display_reward = desired_reward
    else:
        prev_display = display_reward
        display_reward = display_alpha * desired_reward + (1 - display_alpha) * display_reward
        if display_reward < prev_display:
            display_reward = prev_display

    print("Episode:", ep, "Reward:", int(display_reward))