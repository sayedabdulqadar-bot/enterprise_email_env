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

    print STEP LOG

    if result["done"]:
        break