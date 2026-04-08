import pickle

from env import TrafficEnv
from agent import QAgent


env = TrafficEnv()
agent = QAgent()

episodes = 1000

for ep in range(episodes):
    state = env.reset()
    total_reward = 0

    while True:
        action = agent.choose_action(state)
        next_state, reward, done, _ = env.step(action)

        agent.learn(state, action, reward, next_state)

        state = next_state
        total_reward += reward

        if done:
            break

    if ep % 50 == 0:
        print(f"Episode {ep}, Reward: {total_reward}")

with open("qtable.pkl", "wb") as f:
    pickle.dump(agent.q_table, f)
