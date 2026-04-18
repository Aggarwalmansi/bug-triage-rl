# Benchmark Report Template

## Dataset

- Number of issues:
- Repositories:
- Date range:
- Duplicate positive rate:
- Train/validation/test split:
- Human label source:

## Models

- Majority baseline:
- Heuristic baseline:
- Embedding baseline:
- LLM baseline:
- Offline RL policy:

## Metrics

- Task 1: reward, type accuracy, severity accuracy.
- Task 2: reward, precision, recall, F1, false positive rate.
- Task 3: reward, capacity violations.
- Human evaluation: usefulness, evidence quality, calibration.

## Confidence Intervals

Report 95% bootstrap confidence intervals over issue-level or batch-level samples.

## Ablations

- Without comments.
- Without timeline events.
- Without duplicate candidates.
- Without embeddings.
- Without LLM rationale.
- Without offline RL traces.

## Error Analysis

Summarize failures using `docs/error_taxonomy.md`.

## Conclusion

State what the agent can do reliably, what remains unsafe, and what evidence supports the claim.
