import os
import re
from openai import OpenAI

from env.environment import EmailOpsEnv
from env.models import Action

# ---------------- ENV CONFIG ---------------- #

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


def rule_based_agent(email):
    email_lower = email.lower()

    # ---- CLASSIFICATION ---- #
    if "hacked" in email_lower or "login" in email_lower:
        category = "security"
        priority = "urgent"
        route = "security_team"

    elif "refund" in email_lower or "charged" in email_lower or "order" in email_lower:
        category = "billing"
        priority = "high"
        route = "billing_team"

    else:
        category = "general"
        priority = "medium"
        route = "support_team"

    # ---- RESPONSE ---- #
    response = (
        f"We are sorry for the inconvenience. "
        f"Our {route.replace('_', ' ')} will review your issue and assist you shortly."
    )

    entities = extract_entities(email_lower)

    return category, priority, route, response, entities


def model_agent(email):
    if USE_API and client:
        try:
            completion = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {
                        "role": "user",
                        "content": f"Handle this support email:\n{email}\nRespond professionally."
                    }
                ],
                temperature=0.3,
                max_tokens=120
            )
            text = completion.choices[0].message.content
            return text.strip() if text else fallback_response()
        except Exception:
            return fallback_response()
    else:
        return fallback_response()


def fallback_response():
    return "We are sorry for the inconvenience. Our team will assist you shortly."


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

            # ---- AGENT DECISION ---- #
            category, priority, route, base_response, entities = rule_based_agent(email)

            # Optionally enhance response with model
            if USE_API:
                text = model_agent(email)
            else:
                text = base_response

            action = Action(
                category=category,
                priority=priority,
                route=route,
                response=text,
                extracted_entities=entities
            )

            result = env.step(action)

            reward = float(result.get("reward", 0.0))
            done = bool(result.get("done", False))
            error = result.get("error", None)

            rewards.append(reward)

            log_step(step, text, reward, done, error)

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