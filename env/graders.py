def score_entities(pred, truth):
    if not truth:
        return 1.0 if not pred else 0.5
    matches = sum(1 for k in truth if pred.get(k) == truth[k])
    return matches / len(truth)


def score_response_quality(response):
    score = 0
    if "sorry" in response.lower():
        score += 0.3
    if "help" in response.lower():
        score += 0.3
    if len(response) > 20:
        score += 0.4
    return min(score, 1.0)


def grade(action, task):
    score = 0

    if action.category == task["category"]:
        score += 0.2

    if action.priority == task["priority"]:
        score += 0.2

    if action.route == task["route"]:
        score += 0.2

    score += 0.2 * score_entities(action.extracted_entities, task["entities"])
    score += 0.2 * score_response_quality(action.response)

    return min(score, 1.0)