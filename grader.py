def grade(env):
    total = env.lane1 + env.lane2

    # normalize score (lower traffic = better)
    score = max(0, 1 - (total / 150))

    return round(score, 2)
