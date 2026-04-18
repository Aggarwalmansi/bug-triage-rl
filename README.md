---
title: Bug Triage RL Agent
emoji: 🤖
colorFrom: indigo
colorTo: purple
sdk: docker
app_file: server/app.py
pinned: false
---

# Bug Triage RL Benchmark

Bug Triage RL Benchmark is a reinforcement-learning environment for software-engineering decision making. It models issue triage as a sequence of engineering judgments: classify the issue, detect duplicates, identify uncertainty, and allocate limited sprint capacity.

The project is designed to grow from a seed dataset into a serious RL-for-software-engineering benchmark. The current repository includes the environment, reward functions, baseline agents, Groq-powered LLM agent support, and reproducible evaluation scripts.

## Research Motivation

Most issue classifiers optimize static labels. Real triage is harder. Maintainers reason about missing evidence, duplicate reports, subsystem ownership, effort, user impact, regressions, and release risk. A top-tier triage agent should not merely guess labels; it should make calibrated decisions that reduce maintainer load and improve resolution outcomes.

This benchmark frames triage as a sequential decision problem:

1. Observe an issue and repo context.
2. Decide whether the issue is a bug, feature request, docs task, or question.
3. Estimate severity, component, and whether more information is needed.
4. Search previous issues for duplicate candidates.
5. Select sprint work under a capacity budget.
6. Receive reward from correctness, calibration, impact, and planning quality.

## Current Tasks

### Task 1: Triage Classification

The agent predicts:

- `bug_type`: `bug`, `feature`, `question`, or `docs`
- `severity`: `P0` through `P4`
- `component`
- `needs_info`
- optional confidence and rationale

Reward combines type, severity, missing-information detection, and component accuracy.

### Task 2: Duplicate Detection

The agent chooses whether the current issue duplicates a previous issue. Observations include lexical candidate duplicates from issue history. The metric reports precision, recall, F1, accuracy, and a warning when a split has no positive duplicate labels.

### Task 3: Sprint Prioritization

The agent receives a batch of issues and selects work under capacity. Value combines severity, impact, and uncertainty. Reward is normalized against an exact knapsack optimum.

## Architecture

```text
data/issues.json
      |
      v
benchmark.dataset
  - normalization
  - inferred effort and impact
  - duplicate candidate retrieval
  - deterministic train/validation/test split
      |
      v
env.BugTriageEnv
  - OpenEnv-style reset/step/state
  - multi-task observations
  - context and candidate duplicates
      |
      v
agents/
  - majority baseline
  - heuristic baseline
  - Groq triage agent
      |
      v
tasks/
  - task-specific reward functions
      |
      v
benchmark.evaluate
  - reproducible metrics
  - JSON reports
```

## Repository Layout

```text
agents/
  baselines.py          Baseline agents and registry
  embedding_baselines.py Hashing-embedding retrieval and centroid baselines
  groq_agent.py         Groq-backed JSON triage agent
  hybrid_rl_agent.py    Metadata-aware hybrid RL agent with safe action validation
  offline_rl_agent.py   Offline trace policy baseline

benchmark/
  action_validation.py  Validates duplicate IDs, sprint selections, types, and confidence
  audit_dataset.py      Dataset readiness audit
  dataset.py            Dataset normalization, retrieval, splits
  evaluate.py           Evaluation CLI
  embeddings.py         Lightweight text embedding and retrieval index
  metrics.py            Metric helpers
  publish_report.py     Render benchmark JSON reports as Markdown
  report.py             Benchmark report with CIs and ablations
  search.py             Similar issue search CLI

clients/
  groq_client.py        Groq JSON completion wrapper

docs/
  roadmap_status.md
  human_evaluation_rubric.md
  error_taxonomy.md
  benchmark_report_template.md

offline_rl/
  traces.py             Build logged decision traces for offline RL

tasks/
  task1.py              Classification reward
  task2.py              Duplicate reward
  task3.py              Sprint planning reward

app.py                  FastAPI environment server
env.py                  Environment implementation
inference.py            Agent rollout CLI
models.py               Pydantic observation/action/reward schema
```

## Installation

```bash
pip install -r requirements.txt
```

For Groq-backed inference:

```bash
export GROQ_API_KEY="your_key"
export GROQ_MODEL="llama-3.3-70b-versatile"
```

The Groq client uses the official `groq` Python SDK and requests JSON responses from the chat-completions API.

## Run The API

```bash
uvicorn app:app --reload
```

For deployed frontends outside localhost, set:

```bash
export CORS_ALLOW_ORIGINS="https://your-frontend-domain.com"
```

Then inspect:

```bash
curl "http://127.0.0.1:8000/reset?task=task1"
curl "http://127.0.0.1:8000/state"
```

## Run The Frontend

The React dashboard lives in `frontend/` and is not served by FastAPI.

```bash
cd frontend
npm install
npm run dev
```

Open [http://127.0.0.1:5173](http://127.0.0.1:5173).

The frontend resolves its API base in this order:

1. `VITE_API_URL`
2. `http://127.0.0.1:8000` when running locally
3. `https://wahhbhai-bug-triage-env.hf.space` in deployed environments

Example:

```bash
cd frontend
cp .env.example .env
```

The frontend calls the backend directly using backend-native agent APIs:

- `POST /agent/session`
- `POST /agent/train`
- `POST /agent/act`
- `POST /agent/step`
- `GET /agent/state`

The lower-level environment APIs remain available for manual evaluation:

- `GET /reset`
- `POST /step`
- `GET /state`

## Run Baseline Evaluation

```bash
python -m benchmark.evaluate
```

Write a report:

```bash
python -m benchmark.evaluate --output reports/baseline_results.json
```

Evaluate Groq as a baseline:

```bash
python -m benchmark.evaluate --baselines groq-triage
```

Run an environment rollout:

```bash
python inference.py --agent heuristic --task task1
python inference.py --agent groq-triage --task task3
```

Audit whether the dataset is ready for benchmark claims:

```bash
python -m benchmark.audit_dataset
```

Build offline RL traces:

```bash
python -m offline_rl.traces --policy heuristic --output data/decision_traces.jsonl
```

Evaluate embedding and offline-RL baselines:

```bash
python -m benchmark.evaluate --baselines embedding-centroid offline-rl
```

Train and inspect the production hybrid RL agent:

```bash
python inference.py --agent hybrid-rl --task task1 --episodes 25 --show-learning
python inference.py --agent hybrid-rl --task task3 --episodes 25
```

Generate a benchmark report with confidence intervals and ablations:

```bash
python -m benchmark.report --output reports/benchmark_report.json
python -m benchmark.publish_report --input reports/benchmark_report.json --output reports/benchmark_report.md
```

Search for similar issues:

```bash
python -m benchmark.search --issue-id 1 --k 5
```

Build an enriched GitHub dataset with comments and timeline events:

```bash
export GITHUB_TOKEN="your_token"
python scripts/github_ingest.py \
  --repos microsoft/vscode pallets/flask numpy/numpy \
  --target-total 10000 \
  --include-timeline \
  --output data/github_issues_enriched.json
```

## Data Schema

Each issue may contain:

```json
{
  "id": 1,
  "title": "Terminal text is highlighted randomly",
  "body": "Full issue body...",
  "repo": "vscode",
  "labels": ["bug"],
  "type": "bug",
  "severity": "P2",
  "duplicate_of": null,
  "effort": 3,
  "impact": 4,
  "component": "terminal",
  "needs_info": false
}
```

The current seed dataset is small and partially heuristic-labeled. `benchmark.dataset` normalizes missing fields so the environment runs, but serious research claims require a larger curated dataset.

## Training Pipeline

The intended top-tier pipeline is:

1. **Data ingestion**
   Pull GitHub issues, comments, labels, linked PRs, timeline events, duplicate links, milestones, assignees, and close reasons.

2. **Dataset construction**
   Build train/validation/test splits by repo and time. Add duplicate clusters, effort estimates, impact labels, owner/component labels, and final resolution outcomes.

3. **Baselines**
   Evaluate majority, heuristic, retrieval, supervised, LLM, and LLM+retrieval baselines before RL.

4. **Offline policy learning**
   Train from historical maintainer decisions. Actions include classify, ask for info, link duplicate, assign owner, and prioritize.

5. **Reward modeling**
   Learn reward from maintainer corrections, resolution time, duplicate precision, capacity planning value, and human preference judgments.

6. **RL fine-tuning**
   Optimize a policy against the reward model and environment simulator. Use held-out repos and future time slices for evaluation.

7. **Human evaluation**
   Ask maintainers to judge action usefulness, evidence quality, calibration, and whether the recommendation reduces triage burden.
