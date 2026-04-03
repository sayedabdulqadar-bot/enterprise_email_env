from fastapi import FastAPI
from env.environment import EmailOpsEnv
from env.models import Action

app = FastAPI()

env = EmailOpsEnv("easy")


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


@app.get("/")
def root():
    return {"status": "Enterprise Email Env running"}


def main():
    import uvicorn
    uvicorn.run("env.server.app:app", host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()