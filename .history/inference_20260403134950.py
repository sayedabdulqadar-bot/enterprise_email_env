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

    print("[STEP] step=1 action=... reward=0.00 done=false error=null")

    if result["done"]:
        break