# Multi-step reasoning agent (IMPORTANT FOR SCORE)

for step in range(1, 6):

    prompt = f"""
    Email: {obs.email}
    History: {obs.history}

    Return:
    category, priority, route, response, entities(json)
    """

    # call model
    # parse output

    result = env.step(action)

    def log_start(task, env, model):
    print(f"[START] task={task} env={env} model={model}", flush=True)


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

    if result["done"]:
        break