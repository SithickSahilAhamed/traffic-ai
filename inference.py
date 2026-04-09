import os
from openai import OpenAI

from env import TrafficEnv


BASE_URL = os.getenv("API_BASE_URL")
API_KEY = os.getenv("API_KEY")

if BASE_URL and API_KEY:
    client = OpenAI(base_url=BASE_URL, api_key=API_KEY)
else:
    client = None

env = TrafficEnv()


def obs_to_dict(obs):
    if isinstance(obs, dict):
        return obs
    if hasattr(obs, "model_dump"):
        return obs.model_dump()
    if hasattr(obs, "dict"):
        return obs.dict()
    if isinstance(obs, (list, tuple)) and len(obs) >= 4:
        return {
            "lane1": obs[0],
            "lane2": obs[1],
            "light": obs[2],
            "emergency": obs[3],
        }
    raise TypeError(f"Unsupported observation type: {type(obs)!r}")


def get_action(obs):
    obs_data = obs_to_dict(obs)
    if client:
        response = client.chat.completions.create(
            model=os.getenv("MODEL_NAME", "gpt-4o-mini"),
            messages=[
                {
                    "role": "user",
                    "content": (
                        f"lane1={obs_data['lane1']}, lane2={obs_data['lane2']}"
                    ),
                }
            ],
        )
        decision = response.choices[0].message.content.strip()
    else:
        decision = "1" if obs_data["lane2"] > obs_data["lane1"] else "0"

    if "1" in decision:
        return 1
    return 0


def run_episode():
    print("[START] Episode")

    obs = env.reset()
    rewards = []

    for step in range(50):
        action = get_action(obs)

        obs, reward, done, info = env.step(action)
        rewards.append(reward)

        print(
            f"[STEP] step={step} action={action} reward={reward:.2f} done={done} error=null"
        )

    score = max(0, min(1, (100 + sum(rewards)) / 100))

    print(
        f"[END] success={score > 0.8} steps=50 score={score:.2f} rewards={','.join(map(str, rewards))}"
    )


if __name__ == "__main__":
    run_episode()
