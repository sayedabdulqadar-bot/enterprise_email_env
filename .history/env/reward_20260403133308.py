def compute_step_reward(prev_score, new_score, step):
    progress = new_score - prev_score

    # discourage stagnation
    penalty = -0.05 if progress <= 0 else 0

    # discourage long episodes
    step_penalty = -0.01 * step

    return progress + penalty + step_penalty