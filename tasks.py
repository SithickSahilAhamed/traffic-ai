def easy(env):
    return env.lane1 + env.lane2 < 40


def medium(env):
    return env.lane1 + env.lane2 < 25


def hard(env):
    return env.lane1 + env.lane2 < 15
