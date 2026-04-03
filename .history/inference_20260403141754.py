import os
import re
from openai import OpenAI

from env.environment import EmailOpsEnv
from env.models import Action

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
API_KEY = os.getenv("HF_TOKEN")

USE_API = API_KEY is not None

client = None
if USE_API:
    try:
        client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    except Exception:
        client = None
        USE_API = False


# ---------------- LOGGING ---------------- #

def log_start(task, env, model):
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step, action, reward, done, error):
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True
    )


def log_end(success, steps, rewards):
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} rewards={rewards_str}",
        flush=True
    )


# ---------------- SMART AGENT ---------------- #

def extract_entities(email):
    match = re.search(r"#(\d+)", email)
    return {"order_id": match.group(1)} if match else {}


def classify(email):
    e = email.lower()

    if "hacked" in e or "login" in e:
        return "security", "urgent", "security_team"

    elif "refund" in e or "charged" in e or "order" in e:
        return "billing", "high", "billing_team"

    else:
        return "general", "medium", "support_team"


def build_response(step, route):
    # progressively improve response each step
    base = "We are sorry for the inconvenience."

    if step == 1:
        return base

    elif step == 2:
        return base + f" Our {route.replace('_', ' ')} has been notified."

    elif step == 3:
        return base + f" Our {route.replace('_', ' ')} is actively reviewing your issue."

    elif step == 4:
        return base + f" Your issue is being prioritized by our {route.replace('_', ' ')}."

    else:
        return base + f" We are resolving this urgently and will update you shortly."


# ---------------- MAIN TASK ---------------- #

def run_task(task_name):
    env = EmailOpsEnv(task_name)

    rewards = []
    steps = 0
    success = False

    log_start(task_name, "enterprise_email_env", MODEL_NAME)

    try:
        obs = env.reset()

        for step in range(1, 6):
            steps = step

            email = obs.email

            category, priority, route = classify(email)
            entities = extract_entities(email)

            response = build_response(step, route)

            action = Action(
                category=category,
                priority=priority,
                route=route,
                response=response,
                extracted_entities=entities
            )

            result = env.step(action)

            reward = float(result.get("reward", 0.0))
            done = bool(result.get("done", False))
            error = result.get("error", None)

            rewards.append(reward)

            log_step(step, response, reward, done, error)

            if done:
                break

        total_reward = sum(rewards)
        success = total_reward > 0.5

    except Exception as e:
        log_step(steps, "error", 0.0, True, str(e))

    finally:
        log_end(success, steps, rewards)


# ---------------- ENTRY ---------------- #

if __name__ == "__main__":
    for task_name in ["easy", "medium", "hard"]:
        run_task(task_name)