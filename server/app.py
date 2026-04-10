from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import uvicorn

from env import TrafficEnv
from tasks import TASKS, TASK_MAP, get_tasks

app = FastAPI(title="Traffic Signal Control - OpenEnv")

env = TrafficEnv()
current_task_id: Optional[str] = None
episode_rewards: list = []


class ResetRequest(BaseModel):
    task_id: Optional[str] = "easy"

class StepRequest(BaseModel):
    action: int


@app.post("/reset")
def reset(request: ResetRequest = None):
    global current_task_id, episode_rewards

    task_id = (request.task_id if request else None) or "easy"

    if task_id not in TASK_MAP:
        raise HTTPException(400, detail=f"Unknown task_id '{task_id}'. Valid: {list(TASK_MAP)}")

    current_task_id = task_id
    episode_rewards = []

    obs = env.reset(task_id=task_id)
    obs_dict = obs.model_dump() if hasattr(obs, "model_dump") else obs
    return {"observation": obs_dict, "task_id": task_id}


@app.post("/step")
def step(request: StepRequest):
    global episode_rewards

    if request.action not in [0, 1]:
        raise HTTPException(400, detail="Action must be 0 or 1.")

    obs, reward, done, info = env.step(request.action)
    obs_dict = obs.model_dump() if hasattr(obs, "model_dump") else obs
    episode_rewards.append(reward)          # ← collect rewards

    response = {
        "observation": obs_dict,
        "reward": reward,
        "done": done,
        "info": info,
    }

    # On episode end: call grader with the collected reward LIST
    if done and current_task_id and current_task_id in TASK_MAP:
        grader_fn = TASK_MAP[current_task_id]["grader"]   # ← actual callable
        score = grader_fn(episode_rewards)                 # ← pass list
        response["score"] = score
        response["info"]["score"] = score

    return response


@app.get("/state")
def state():
    return {"state": env.state()}


@app.get("/tasks")
def tasks_endpoint():
    return {"tasks": get_tasks()}


@app.post("/grade")
def grade(request: ResetRequest):
    task_id = request.task_id or "easy"

    if task_id not in TASK_MAP:
        raise HTTPException(400, detail=f"Unknown task_id '{task_id}'")

    task = TASK_MAP[task_id]
    obs = env.reset(task_id=task_id)
    rewards = []
    done = False

    while not done:
        # Convert Observation object to dict safely
        obs_dict = obs.model_dump() if hasattr(obs, "model_dump") else dict(obs)

        if obs_dict.get("emergency") == 1:
            action = 1
        else:
            action = 0 if obs_dict.get("lane1", 0) >= obs_dict.get("lane2", 0) else 1

        obs, reward, done, info = env.step(action)
        rewards.append(reward)

    score = task["grader"](rewards)

    return {
        "task_id": task_id,
        "score": score,
        "total_reward": sum(rewards),
        "steps": len(rewards),
    }


@app.get("/health")
def health():
    return {"status": "ok"}


def main():
    uvicorn.run(app, host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()
