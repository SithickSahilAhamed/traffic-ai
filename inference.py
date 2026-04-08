import random

from env import TrafficEnv
from grader import grade
from models import Action

TASK_NAME = "easy"
BENCHMARK = "traffic-control"
MODEL_NAME = "baseline-agent"
MAX_STEPS = 50


def log_start():
    print(f"[START] task={TASK_NAME} env={BENCHMARK} model={MODEL_NAME}")


def log_step(step, action, reward, done):
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} error=null"
    )


def log_end(success, steps, score, rewards):
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.2f} rewards={rewards_str}")


def run_episode():
    env = TrafficEnv()
    obs = env.reset()

    rewards = []
    total_reward = 0

    log_start()

    for step in range(1, MAX_STEPS + 1):
        # simple intelligent policy
        desired_light = 0 if env.lane1 > env.lane2 else 1
        if env.light != desired_light:
            action = 1  # switch
        else:
            action = 0  # stay

        obs, reward, done, _ = env.step(Action(action=action))

        rewards.append(reward)
        total_reward += reward

        log_step(step, action, reward, done)

        if done:
            break

    score = grade(env)
    success = score > 0.5

    log_end(success, step, score, rewards)


if __name__ == "__main__":
    run_episode()
