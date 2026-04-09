import os
import json

from openai import OpenAI
from env import TrafficEnv

client = OpenAI(
    base_url=os.environ.get("API_BASE_URL"),
    api_key=os.environ.get("API_KEY")
)


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


def get_action(obs, strategy="balanced"):
    obs_data = obs_to_dict(obs)

    if strategy == "emergency_priority":
        prompt = f"""
Traffic Control - Emergency Priority Mode:

Lane1 vehicles: {obs_data['lane1']}
Lane2 vehicles: {obs_data['lane2']}
Emergency vehicle present: {obs_data['emergency']}

Rules: If emergency=1, always prioritize the lane with the emergency vehicle.
Otherwise choose the lane with more vehicles.

Choose:
0 -> Lane1 green
1 -> Lane2 green

Answer only 0 or 1.
"""
    elif strategy == "balanced":
        prompt = f"""
Traffic Control - Balanced Mode:

Lane1 vehicles: {obs_data['lane1']}
Lane2 vehicles: {obs_data['lane2']}
Emergency: {obs_data['emergency']}

Choose the action that minimizes total wait time across both lanes.

Choose:
0 -> Lane1 green
1 -> Lane2 green

Answer only 0 or 1.
"""
    else:  # greedy
        prompt = f"""
Traffic Control - Greedy Mode:

Lane1: {obs_data['lane1']}
Lane2: {obs_data['lane2']}
Emergency: {obs_data['emergency']}

Always pick the lane with more vehicles.

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


def clamp_score(score: float) -> float:
    """Score MUST be strictly between 0 and 1 (never 0.0 or 1.0 exactly)."""
    return max(0.001, min(0.999, float(score)))


def run_episode(task_id: str, strategy: str, steps: int = 50, grader: str = "grader.grade") -> dict:
    print(f"[START] task={task_id} strategy={strategy}")

    env = TrafficEnv()
    obs = env.reset()
    rewards = []

    for step in range(steps):
        action = get_action(obs, strategy=strategy)
        obs, reward, done, info = env.step(action)
        rewards.append(reward)
        print(f"[STEP] task={task_id} step={step} action={action} reward={reward:.2f} done={done}")
        if done:
            break

    total_reward = sum(rewards)
    raw_score = (100 + total_reward) / 200  # normalized to ~0–1 range
    score = clamp_score(raw_score)

    print(f"[END] task={task_id} score={score:.4f} total_reward={total_reward:.2f}")

    return {
        "task_id": task_id,
        "grader": grader,
        "score": score,
    }


if __name__ == "__main__":
    results = [
        run_episode("easy",   strategy="balanced",          steps=50),
        run_episode("medium", strategy="emergency_priority", steps=50),
        run_episode("hard",   strategy="greedy",            steps=50),
    ]

    # This JSON output is what the HuggingFace evaluator parses
    print(json.dumps(results))