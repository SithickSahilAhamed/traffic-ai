"""
grader.py
---------
Each grader receives a LIST of reward floats (not a dict).
Score must be strictly in (0, 1) — never 0.0 or 1.0.
"""


def normalize(score: float) -> float:
    """Clamp to strict open interval (0, 1)."""
    if score <= 0.0:
        return 0.01
    if score >= 1.0:
        return 0.99
    return round(score, 6)


def compute_score(rewards: list) -> float:
    """
    Args:
        rewards: list of float rewards from the episode   ← LIST not dict
    Returns:
        float strictly in (0, 1)
    """
    if not rewards:
        return 0.5  # safe midpoint default

    avg_reward = sum(rewards) / len(rewards)

    # avg_reward is negative (congestion penalty); abs makes it positive
    # 1 / (1 + x) maps (0, inf) → (0, 1) but never hits boundaries
    score = 1.0 / (1.0 + abs(avg_reward))

    return normalize(score)


def grade_easy(rewards: list) -> float:
    return compute_score(rewards)


def grade_medium(rewards: list) -> float:
    return compute_score(rewards)


def grade_hard(rewards: list) -> float:
    return compute_score(rewards)
