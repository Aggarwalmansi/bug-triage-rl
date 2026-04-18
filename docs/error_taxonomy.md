# Error Taxonomy

Use this taxonomy when analyzing failed triage decisions.

## Classification Errors

- `wrong_type`: bug, feature, question, or docs classification is wrong.
- `wrong_severity`: severity is too high or too low.
- `wrong_component`: subsystem/component is wrong.
- `missed_needs_info`: agent should have asked for more information.
- `unnecessary_needs_info`: agent asks for info despite enough evidence.

## Duplicate Errors

- `false_duplicate`: agent links unrelated issues.
- `missed_duplicate`: agent fails to identify a true duplicate.
- `wrong_duplicate_target`: duplicate exists but the linked target is not the canonical issue.
- `weak_duplicate_evidence`: duplicate decision lacks useful comparison.

## Planning Errors

- `capacity_violation`: selected work exceeds capacity.
- `low_value_selection`: selected low-impact work over high-impact alternatives.
- `effort_blind_selection`: ignores effort or complexity.
- `blocked_work_selected`: selects work that lacks required information or dependencies.
- `release_risk_missed`: misses regression, security, or release-blocking risk.

## Reasoning Errors

- `unsupported_claim`: rationale claims facts not in the issue context.
- `missing_key_evidence`: rationale ignores important issue details.
- `overconfidence`: confidence too high for ambiguous evidence.
- `underconfidence`: confidence too low for clear evidence.
- `repo_norm_mismatch`: decision ignores repo-specific labeling or triage conventions.

## Evaluation Errors

- `label_noise`: gold label appears wrong or heuristic-generated.
- `ambiguous_gold`: multiple reasonable labels exist.
- `insufficient_context`: issue cannot be judged without comments, linked PRs, logs, or history.
