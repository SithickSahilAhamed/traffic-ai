import matplotlib.pyplot as plt

# dummy rewards for demo
rewards = [-5000, -4200, -3500, -2800, -2000]

plt.plot(rewards)
plt.title("Training Improvement")
plt.xlabel("Episodes")
plt.ylabel("Reward")
plt.show()
