import os
from openai import OpenAI

# Correct import (must match your environment class)
from env.environment import EmailOpsEnv
from env.models import Action

# ---------------- ENV CONFIG ---------------- #

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
API_KEY = os.getenv("HF_TOKEN")

USE_API = API_KEY is not None

# Initialize client safely
client = None
if USE_API:
    try:
        client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    except Exception:
        client = None
        USE_API = False


# ---------------- LOGGING (STRICT FORMAT) ---------------- #

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


# ---------------- MODEL CALL (SAFE) ---------------- #

def get_model_response(email_text):
    """
    Safe model call with fallback.
    Never crashes.
    """
    if USE_API and client is not None:
        try:
            completion = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {
                        "role": "user",
                        "content": f"Handle this support email:\n{email_text}\nRespond appropriately."
                    }
                ],
                temperature=0.3,
                max_tokens=100
            )
            text = completion.choices[0].message.content
            return text.strip() if text else fallback_response()
        except Exception:
            return fallback_response()
    else:
        return fallback_response()


def fallback_response():
    return "Sorry for the issue. We will help resolve this as soon as possible."


# ---------------- TASK RUNNER ---------------- #

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

            # Safe response generation
            text = get_model_response(obs.email)

            # Deterministic safe action
            action = Action(
                category="billing",
                priority="high",
                route="billing_team",
                response=text,
                extracted_entities={}
            )

            result = env.step(action)

            reward = float(result.get("reward", 0.0))
            done = bool(result.get("done", False))
            error = result.get("error", None)

            rewards.append(reward)

            log_step(step, text, reward, done, error)

            if done:
                break

        # Normalize success
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