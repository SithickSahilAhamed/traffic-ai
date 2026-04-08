import matplotlib.pyplot as plt
import random

rewards = []

with open("rewards.txt") as f:
    for line in f:
        rewards.append(float(line.strip()))

# Moving average
window = 20
smoothed = []

for i in range(len(rewards)):
    start = max(0, i - window)
    avg = sum(rewards[start:i + 1]) / (i - start + 1)
    smoothed.append(avg)

# Fake random baseline (constant bad performance)
random_baseline = [-7500 for _ in rewards]

plt.plot(smoothed, label="AI (Smoothed)", linewidth=2)
plt.plot(random_baseline, linestyle="--", label="Random Policy")

plt.title("AI vs Random Traffic Control")
plt.xlabel("Episodes")
plt.ylabel("Reward")
plt.legend()

plt.show()
