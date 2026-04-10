---
title: Traffic AI OpenEnv
emoji: 🚦
colorFrom: red
colorTo: yellow
sdk: docker
pinned: false
tags:
  - openenv
---

# 🚦 Traffic Signal Control — OpenEnv Environment

An AI-based traffic signal control system simulating a 4-way junction.
An LLM agent decides which lane receives the green signal at each step
to minimize congestion across three difficulty levels.

---

## 🌍 Environment Description

Real-world motivation: Traffic signal control is a critical urban infrastructure problem.
Poor signal timing causes congestion, fuel waste, and emergency vehicle delays.
This environment simulates a 2-lane junction where an AI agent must make
real-time decisions to keep traffic flowing efficiently.

The agent observes queue lengths, the current signal state, and whether an
emergency vehicle is present — then decides which lane gets the green light.

---

## 📐 Observation Space

| Field | Type | Description |
|---|---|---|
| `lane1` | int | Number of vehicles queued in Lane 1 |
| `lane2` | int | Number of vehicles queued in Lane 2 |
| `light` | int (0 or 1) | Current green light (0 = Lane 1, 1 = Lane 2) |
| `emergency` | int (0 or 1) | Emergency vehicle present (1 = yes) |

## ⚡ Action Space

| Action | Meaning |
|---|---|
| `0` | Give green signal to Lane 1 |
| `1` | Give green signal to Lane 2 |

## 🏆 Reward Function

```
reward = -(lane1 + lane2)          # penalize total congestion
       + 10 if emergency handled correctly
       - 10 if emergency ignored
```

Rewards are dense (every step), providing continuous signal throughout the episode.

---

## 📋 Tasks

| Task | Steps | Emergency Prob | Arrival Rate | Difficulty |
|---|---|---|---|---|
| `easy` | 20 | 0% | 0–2 cars/step | Low traffic, no emergencies |
| `medium` | 30 | 10% | 0–4 cars/step | Uneven traffic, occasional emergencies |
| `hard` | 40 | 25% | 0–7 cars/step | Heavy traffic, frequent emergencies |

### Expected Baseline Scores

| Task | Baseline Score | Notes |
|---|---|---|
| `easy` | ~0.72 | Heuristic agent performs well |
| `medium` | ~0.61 | Emergencies add difficulty |
| `hard` | ~0.48 | Heavy load challenges any agent |

---

## 🤖 Agent Design (inference.py)

The agent uses a 6-layer hybrid decision stack:

```
1. Emergency override      → immediate response
2. Starvation prevention   → fairness across lanes
3. Heavy imbalance rule    → fast deterministic path
4. Value function (pseudo-RL) → learned signal
5. LLM decision            → ambiguous cases only
6. Anti-oscillation lock   → stability control
```

LLM is called **only when the decision is genuinely ambiguous** —
this avoids unnecessary API calls and reduces latency.

---

## 🚀 Setup & Usage

### Environment Variables Required

```bash
API_BASE_URL=<provided by hackathon>
API_KEY=<provided by hackathon>
MODEL_NAME=gpt-4o-mini
```

### Run Locally

```bash
pip install fastapi uvicorn openai pydantic openenv-core numpy

# Start the server
uvicorn app:app --host 0.0.0.0 --port 7860

# Run inference (in another terminal)
python inference.py
```

### Docker

```bash
docker build -t traffic-ai .
docker run -p 7860:7860 \
  -e API_BASE_URL=... \
  -e API_KEY=... \
  traffic-ai
```

---

## 🔌 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| POST | `/reset` | Reset environment, pass `{"task_id": "easy"}` |
| POST | `/step` | Take action, pass `{"action": 0}` or `{"action": 1}` |
| GET | `/state` | Get current environment state |
| GET | `/tasks` | List all 3 tasks |
| POST | `/grade` | Run full episode and return graded score |
| GET | `/health` | Health check |

---

## 📁 Project Structure

```
traffic-ai/
├── app.py          # FastAPI server (OpenEnv endpoints)
├── env.py          # TrafficEnv — core simulation logic
├── inference.py    # Hybrid LLM agent — runs all 3 tasks
├── grader.py       # Score functions for each task
├── tasks.py        # Task definitions with grader linkage
├── models.py       # Pydantic models (Observation, Action)
├── openenv.yaml    # OpenEnv spec
├── Dockerfile      # Container setup
└── server/
    └── app.py      # Mirror of app.py (multi-mode support)
```

---

## 📊 Grader Design

Scores are computed using `1 / (1 + |avg_reward|)` normalized through
a sigmoid function — mathematically guaranteed to stay strictly in (0, 1).
Boundary values 0.0 and 1.0 are impossible by design.
