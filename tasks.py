from grader import grade_easy, grade_medium, grade_hard  # actual callables, NOT strings

TASKS = [
    {
        "id": "easy",
        "description": "Low traffic balancing: minimize congestion over 20 steps with no emergencies.",
        "grader": grade_easy,
    },
    {
        "id": "medium",
        "description": "Uneven traffic handling: balance lane queues over 30 steps with occasional emergencies.",
        "grader": grade_medium,
    },
    {
        "id": "hard",
        "description": "Emergency + traffic optimization: heavy traffic and frequent emergencies over 40 steps.",
        "grader": grade_hard,
    },
]

TASK_MAP = {t["id"]: t for t in TASKS}


def get_tasks():
    """Serializable task list (no grader function)."""
    return [{"id": t["id"], "description": t["description"]} for t in TASKS]
