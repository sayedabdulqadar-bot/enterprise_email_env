import os
from openai import OpenAI
from env.environment import EmailEnv
from env.models import Action

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
API_KEY = os.getenv("HF_TOKEN")

client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)


def log_start(task, env, model):
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step, action, reward, done, error):
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} error={error or 'null'}",
        flush=True
    )


def log_end(success, steps, rewards):
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} rewards={rewards_str}", flush=True)


def run_task(task_name):
    env = EmailEnv(task_name)

    obs = env.reset()   # ✅ FIXED

    rewards = []
    steps = 0

    log_start(task_name, "enterprise_email_env", MODEL_NAME)

    for step in range(1, 6):
        steps = step

        # simple baseline action (safe default)
        action = Action(
            category="billing",
            priority="high",
            route="billing_team",
            response="Sorry for the issue, we will help you resolve this.",
            extracted_entities={}
        )

        result = env.step(action)

        reward = result["reward"]
        done = result["done"]

        rewards.append(reward)

        log_step(step, action.response, reward, done, None)

        if done:
            break   # ✅ ONLY PLACE break EXISTS

    success = sum(rewards) > 0.5
    log_end(success, steps, rewards)


if __name__ == "__main__":
    for task in ["easy", "medium", "hard"]:
        run_task(task)