from fastapi import FastAPI
from env.environment import EmailOpsEnv
from env.models import Action

app = FastAPI()

# initialize env
env = EmailOpsEnv("easy")


@app.get("/")
def root():
    return {"status": "ok", "message": "Enterprise Email Env running"}


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
        result = env.step(action_obj)
        return result
    except Exception as e:
        return {
            "observation": {},
            "reward": 0.0,
            "done": True,
            "info": {"error": str(e)}
        }


# OPTIONAL: health endpoint (helps HF)
@app.get("/health")
def health():
    return {"status": "healthy"}