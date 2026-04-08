from fastapi import FastAPI
from env import TrafficEnv

app = FastAPI()
env = TrafficEnv()


@app.post("/reset")
def reset():
    obs = env.reset()
    return {"observation": obs}


@app.post("/step")
def step(action: dict):
    obs, reward, done, info = env.step(action["action"])
    return {
        "observation": obs,
        "reward": reward,
        "done": done,
        "info": info,
    }
