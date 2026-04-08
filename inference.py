import os

from openai import OpenAI

from env import TrafficEnv


client = OpenAI(
    base_url=os.environ.get("API_BASE_URL"),
    api_key=os.environ.get("API_KEY")
)

env = TrafficEnv()


def get_action(obs):
    prompt = f"""
Traffic Control Decision:

Lane1: {obs['lane1']}
Lane2: {obs['lane2']}
Emergency: {obs['emergency']}

Choose:
0 6 Lane1 green
1 6 Lane2 green

Answer only 0 or 1.
"""

    response = client.chat.completions.create(
        model=os.environ.get("MODEL_NAME"),
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

    return int(response.choices[0].message.content.strip())


def run_episode():
    print("[START] Episode")

    obs = env.reset()
    rewards = []

    for step in range(50):
        action = get_action(obs)

        obs, reward, done, info = env.step(action)
        rewards.append(reward)

        print(f"[STEP] step={step} action={action} reward={reward:.2f} done={done} error=null")

    score = max(0, min(1, (100 + sum(rewards)) / 100))

    print(
        f"[END] success={score > 0.8} steps=50 score={score:.2f} rewards={','.join(map(str, rewards))}"
    )


if __name__ == "__main__":
    run_episode()
