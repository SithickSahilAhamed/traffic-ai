import os

from openai import OpenAI

from env import TrafficEnv


client = OpenAI(
    base_url=os.environ.get("API_BASE_URL"),
    api_key=os.environ.get("API_KEY")
)

env = TrafficEnv()


def obs_to_dict(obs):
    if isinstance(obs, dict):
        return obs
    if hasattr(obs, "model_dump"):
        return obs.model_dump()
    if hasattr(obs, "dict"):
        return obs.dict()
    return {
        "lane1": getattr(obs, "lane1", 0),
        "lane2": getattr(obs, "lane2", 0),
        "light": getattr(obs, "light", 0),
        "emergency": getattr(obs, "emergency", 0),
    }


def get_action(obs):
    obs_data = obs_to_dict(obs)
    prompt = f"""
Traffic Control Decision:

Lane1: {obs_data['lane1']}
Lane2: {obs_data['lane2']}
Emergency: {obs_data['emergency']}

Choose:
0 -> Lane1 green
1 -> Lane2 green

Answer only 0 or 1.
"""

    try:
        response = client.chat.completions.create(
            model=os.environ.get("MODEL_NAME"),
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        content = response.choices[0].message.content.strip()
        for ch in content:
            if ch in ("0", "1"):
                return int(ch)
    except Exception as exc:
        print(f"[ERROR] action_error={exc}")

    return 0


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
