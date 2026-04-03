from fastapi import FastAPI
from env.environment import EmailOpsEnv
from env.models import Action

app = FastAPI()

env = EmailOpsEnv("easy")


@app.get("/")
def root():
    return {"status": "ok"}


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/reset")
def reset():
    obs = env.reset()
    return {
        "observation": obs.dict() if hasattr(obs, "dict") else obs,
        "reward": 0.0,
        "done": False,
        "info": {}
    }


@app.post("/step")
def step(action: dict):
    try:
        action_obj = Action(**action)
        return env.step(action_obj)
    except Exception as e:
        return {
            "observation": {},
            "reward": 0.0,
            "done": True,
            "info": {"error": str(e)}
        }