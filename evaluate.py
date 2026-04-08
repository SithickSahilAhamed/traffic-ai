from env import TrafficEnv
from agent import QAgent
import random


def run_random(env):
    state = env.reset()
    total = 0
    while True:
        action = random.randint(0, 1)
        state, reward, done, _ = env.step(action)
        total += reward
        if done:
            return total


def run_ai(env, agent):
    state = env.reset()
    total = 0
    while True:
        action = agent.choose_action(state)
        state, reward, done, _ = env.step(action)
        total += reward
        if done:
            return total


env = TrafficEnv()
agent = QAgent()

# assume trained agent already used

random_score = run_random(env)
ai_score = run_ai(env, agent)

print("Random:", random_score)
print("AI:", ai_score)
