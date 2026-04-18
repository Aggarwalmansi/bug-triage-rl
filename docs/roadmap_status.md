# Roadmap Status

## Implemented

- 10k+ dataset ingestion pipeline in `scripts/github_ingest.py`.
- Comment ingestion for GitHub issues.
- Optional timeline-event ingestion for labels, assignments, closure events, and duplicate signals.
- Duplicate reference parsing from issue bodies and comments.
- Maintainer-label duplicate signal parsing from timeline events.
- Lightweight embedding retrieval in `benchmark/embeddings.py`.
- Similar issue search CLI in `benchmark/search.py`.
- Supervised embedding baseline in `agents/embedding_baselines.py`.
- Production hybrid RL agent in `agents/hybrid_rl_agent.py`.
- Action validation in `benchmark/action_validation.py` and `env.step`.
- Offline RL trace generation in `offline_rl/traces.py`.
- Offline trace policy baseline in `agents/offline_rl_agent.py`.
- Human evaluation rubric in `docs/human_evaluation_rubric.md`.
- Error taxonomy in `docs/error_taxonomy.md`.
- Benchmark report template in `docs/benchmark_report_template.md`.
- Automated benchmark report with bootstrap confidence intervals and ablations in `benchmark/report.py`.
- Markdown report publisher in `benchmark/publish_report.py`.

## Runnable Now On The Seed Dataset

```bash
python -m benchmark.audit_dataset
python -m benchmark.evaluate --baselines majority heuristic embedding-centroid offline-rl
python inference.py --agent hybrid-rl --task task1 --episodes 25 --show-learning
python -m offline_rl.traces --policy heuristic --output data/decision_traces.jsonl
python -m benchmark.report --output reports/benchmark_report.json
python -m benchmark.publish_report --input reports/benchmark_report.json --output reports/benchmark_report.md
python -m benchmark.search --issue-id 1 --k 5
```

## Requires External Data Or Credentials

- Actually expanding to 10k+ issues requires GitHub API access and should use `GITHUB_TOKEN`.
- High-quality duplicate clusters require repos with duplicate labels/comments plus validation.
- Production-grade embedding retrieval should use a stronger embedding model once dependencies and scale are approved.
- Serious offline RL requires real historical maintainer decision traces, not synthetic traces generated from baselines.
- Publishable benchmark claims require human label validation and human evaluation.

## Current Dataset Blockers

- The seed dataset has only 22 issues.
- It has zero positive duplicate examples.
- Labels are seed-scale and partly heuristic.
- The local results are useful smoke tests, not research evidence.
