def grade(episode_result):
    rewards = episode_result.get("rewards", [])

    if not rewards:
        return 0.1

    total_reward = sum(rewards)

    # Safer normalization
    score = 1 / (1 + abs(total_reward) / 100)

    # Force inside (0,1)
    score = max(0.01, min(0.99, score))

    return float(score)
