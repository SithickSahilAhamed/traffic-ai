from fastapi import FastAPI
import uvicorn
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
        "info": info
    }


def main():
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()
