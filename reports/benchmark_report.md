# Bug Triage RL Benchmark Report

## Dataset Audit

- Issues: 22
- Duplicate positive rate: 0.000
- Repositories: {'vscode': 8, 'flask': 11, 'numpy': 3}

### Warnings

- Dataset is too small for research-grade claims.
- No positive duplicate labels; duplicate accuracy is misleading.
- Labels appear seed-scale; validate against human or maintainer decisions before claiming learning.

## Results

### majority
- task1: reward=0.550, bootstrap=0.550 [0.475, 0.625]
- task2: reward=1.000, bootstrap=1.000 [1.000, 1.000]
- task3: reward=0.722, bootstrap=0.722 [0.722, 0.722]

### heuristic
- task1: reward=1.000, bootstrap=1.000 [1.000, 1.000]
- task2: reward=1.000, bootstrap=1.000 [1.000, 1.000]
- task3: reward=1.000, bootstrap=1.000 [1.000, 1.000]

### embedding-centroid
- task1: reward=0.550, bootstrap=0.550 [0.288, 0.725]
- task2: reward=1.000, bootstrap=1.000 [1.000, 1.000]
- task3: reward=1.000, bootstrap=1.000 [1.000, 1.000]

### offline-rl
- task1: reward=0.512, bootstrap=0.512 [0.200, 0.825]
- task2: reward=1.000, bootstrap=1.000 [1.000, 1.000]
- task3: reward=1.000, bootstrap=1.000 [1.000, 1.000]

## Ablations

- full: task1_reward=0.550, task3_reward=1.000
- no_comments: task1_reward=0.550, task3_reward=1.000
- no_labels: task1_reward=0.550, task3_reward=1.000
- title_only: task1_reward=0.625, task3_reward=1.000

## Human Evaluation

- Rubric: docs/human_evaluation_rubric.md
- Error taxonomy: docs/error_taxonomy.md
- Template: docs/benchmark_report_template.md

## Interpretation

This report is valid as a pipeline smoke test on the seed dataset. It is not sufficient for research-grade claims until the enriched GitHub dataset contains thousands of issues, positive duplicate clusters, and human or maintainer-validated labels.
