def grade(episode_result):
    rewards = episode_result.get("rewards", [])

    if not rewards:
        return 0.1  # avoid zero

    total_reward = sum(rewards)

    # Normalize score into (0,1)
    score = (100 + total_reward) / 200

    # Clamp STRICTLY between (0,1)
    if score <= 0:
        score = 0.01
    elif score >= 1:
        score = 0.99

    return score
