"""
inference.py — Hybrid (Rule + Value + LLM) Traffic Signal Controller
- Deadline build: competition-ready
- Runs all 3 tasks, uses registered graders, strict (0,1) score
"""

import os
from openai import OpenAI
from env import TrafficEnv
from tasks import TASKS, TASK_MAP

# ── LLM Client (hackathon proxy) ──────────────────────────────────────────────
client = None
_base_url = os.getenv("API_BASE_URL")
_api_key  = os.getenv("API_KEY")
if _base_url and _api_key:
    client = OpenAI(base_url=_base_url, api_key=_api_key)

# ── Agent memory (reset per episode) ─────────────────────────────────────────
class AgentMemory:
    def __init__(self):
        self.last_action      = 0
        self.same_count       = 0
        self.lane1_cumulative = 0
        self.lane2_cumulative = 0
        self.switch_count     = 0
        self.step             = 0

    def update(self, action, l1, l2):
        self.lane1_cumulative += l1
        self.lane2_cumulative += l2
        if action != self.last_action:
            self.switch_count += 1
            self.same_count = 0
        else:
            self.same_count += 1
        self.last_action = action
        self.step += 1


# ── Value estimator (pseudo-RL) ───────────────────────────────────────────────
def compute_value(l1: int, l2: int, action: int) -> float:
    """
    Estimate value of an action.
    Rewards clearing the bigger lane while penalizing neglect of the other.
    """
    if action == 0:
        return (l1 * 1.2) - (l2 * 0.6)
    else:
        return (l2 * 1.2) - (l1 * 0.6)


# ── LLM decision (called only for ambiguous cases) ───────────────────────────
def llm_decision(obs: dict, task_description: str, memory: AgentMemory) -> int:
    if not client:
        return 0 if obs["lane1"] >= obs["lane2"] else 1

    prompt = f"""You are an expert AI traffic signal controller.

Task: {task_description}

Current State:
- Lane 1 queue: {obs['lane1']} vehicles
- Lane 2 queue: {obs['lane2']} vehicles
- Current green light: Lane {'1' if obs['light'] == 0 else '2'}
- Emergency vehicle: {'YES — override required' if obs['emergency'] else 'No'}

Context:
- Step: {memory.step}
- Lane 1 total wait so far: {memory.lane1_cumulative}
- Lane 2 total wait so far: {memory.lane2_cumulative}
- Switches made: {memory.switch_count}

Strategy rules:
1. If emergency vehicle present → always prioritize the more congested lane
2. Avoid switching too frequently — switching has a penalty
3. Prevent starvation — don't ignore one lane for too long
4. Minimize total vehicles waiting across both lanes

Think briefly, then respond with ONLY a single digit: 0 or 1.
0 = green for Lane 1
1 = green for Lane 2"""

    try:
        response = client.chat.completions.create(
            model=os.getenv("MODEL_NAME", "gpt-4o-mini"),
            messages=[{"role": "user", "content": prompt}],
            max_tokens=10,
            temperature=0,
        )
        raw = response.choices[0].message.content.strip()
        action = int(raw[0])
        if action not in [0, 1]:
            raise ValueError
        return action
    except Exception as e:
        print(f"  [WARN] LLM failed ({e}), using heuristic fallback.")
        return 0 if obs["lane1"] >= obs["lane2"] else 1


# ── Main decision engine ──────────────────────────────────────────────────────
def get_action(obs: dict, task_description: str, memory: AgentMemory) -> tuple:
    l1        = obs["lane1"]
    l2        = obs["lane2"]
    emergency = obs["emergency"]
    reason    = ""

    # LAYER 1: Emergency hard override
    if emergency == 1:
        action = 1  # lane2 priority for emergency (consistent with env.py)
        reason = "emergency_override"

    # LAYER 2: Starvation prevention (one lane ignored too long)
    elif memory.lane1_cumulative > memory.lane2_cumulative * 2.0:
        action = 0
        reason = "starvation_prevention_lane1"
    elif memory.lane2_cumulative > memory.lane1_cumulative * 2.0:
        action = 1
        reason = "starvation_prevention_lane2"

    # LAYER 3: Heavy imbalance — fast deterministic path
    elif abs(l1 - l2) > 8:
        action = 0 if l1 > l2 else 1
        reason = "heavy_imbalance"

    # LAYER 4: Value-based decision (pseudo-RL)
    else:
        v0 = compute_value(l1, l2, 0)
        v1 = compute_value(l1, l2, 1)
        if abs(v0 - v1) > 3:
            action = 0 if v0 > v1 else 1
            reason = f"value_based(v0={v0:.1f},v1={v1:.1f})"
        else:
            # LAYER 5: LLM for genuinely ambiguous cases
            action = llm_decision(obs, task_description, memory)
            reason = "llm_decision"

    # LAYER 6: Anti-oscillation lock
    # Don't switch unless held current long enough or strong reason
    if reason not in ("emergency_override", "starvation_prevention_lane1",
                      "starvation_prevention_lane2"):
        if action != memory.last_action and memory.same_count < 2:
            action = memory.last_action
            reason += "+stability_lock"

    memory.update(action, l1, l2)
    return action, reason


# ── Episode runner ────────────────────────────────────────────────────────────
def run_episode(task_id: str = "easy") -> dict:
    task   = TASK_MAP[task_id]
    memory = AgentMemory()
    env    = TrafficEnv()

    print(f"\n[START] task={task_id} | {task['description']}")

    obs     = env.reset(task_id=task_id)
    rewards = []
    step    = 0

    while True:
        obs_dict = obs if isinstance(obs, dict) else obs.model_dump()
        action, reason = get_action(obs_dict, task["description"], memory)

        obs, reward, done, info = env.step(action)
        rewards.append(reward)

        print(
            f"  [STEP] {step+1:02d} | action={action} reason={reason} | "
            f"lane1={obs_dict['lane1']} lane2={obs_dict['lane2']} "
            f"emergency={obs_dict['emergency']} | reward={reward:.2f}"
        )

        step += 1
        if done:
            break

    # Use registered grader — guaranteed (0,1) score
    score        = task["grader"](rewards)
    total_reward = sum(rewards)

    print(
        f"[END] task={task_id} steps={step} "
        f"total_reward={total_reward:.2f} score={score:.4f} "
        f"switches={memory.switch_count}"
    )

    return {
        "task_id":      task_id,
        "steps":        step,
        "total_reward": total_reward,
        "score":        score,
        "switches":     memory.switch_count,
    }


# ── Entry point: run ALL 3 tasks ──────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 65)
    print("Traffic Signal Control — Hybrid Agent (Rule + Value + LLM)")
    print("=" * 65)

    results = []
    for task in TASKS:
        result = run_episode(task["id"])
        results.append(result)

    print("\n" + "=" * 65)
    print("FINAL SUMMARY")
    print("=" * 65)
    for r in results:
        print(
            f"  {r['task_id']:8s} | score={r['score']:.4f} | "
            f"reward={r['total_reward']:.1f} | "
            f"steps={r['steps']} | switches={r['switches']}"
        )
