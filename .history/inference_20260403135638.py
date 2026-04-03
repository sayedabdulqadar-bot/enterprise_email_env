def run_task(task_name):
    env = EmailEnv(task_name)
    obs = env.reset()

    rewards = []
    steps = 0

    log_start(task_name, "enterprise_email_env", MODEL_NAME)

    for step in range(1, 6):
        steps = step

        # dummy action (replace with model later)
        action = Action(
            category="billing",
            priority="high",
            route="billing_team",
            response="Sorry for the issue, we will help you.",
            extracted_entities={}
        )

        result = env.step(action)

        reward = result["reward"]
        done = result["done"]

        rewards.append(reward)

        log_step(step, action.response, reward, done, None)

        if done:
            break

    success = sum(rewards) > 0.5
    log_end(success, steps, rewards)