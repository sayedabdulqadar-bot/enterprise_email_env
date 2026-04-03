def compute_step_reward(prev_score, new_score, step):
    """
    Improved reward function:
    - Encourages progress
    - Doesn't punish small variations
    - Rewards consistency
    """

    # positive progress
    progress = new_score - prev_score

    # small tolerance (avoid negative noise)
    if progress < 0 and abs(progress) < 0.05:
        progress = 0.02  # small positive reward

    # reward staying good
    if new_score > 0.7:
        bonus = 0.05
    else:
        bonus = 0

    # slight step penalty (to encourage efficiency)
    step_penalty = -0.01 * step

    reward = progress + bonus + step_penalty

    return max(min(reward, 1.0), -1.0)