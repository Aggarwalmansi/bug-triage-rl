# Human Evaluation Rubric

Use this rubric when maintainers or senior engineers judge agent triage decisions.

Each item is scored from 1 to 5.

## Decision Usefulness

1. Harmful or misleading.
2. Mostly unhelpful; requires substantial correction.
3. Partially useful but incomplete.
4. Useful with minor corrections.
5. Ready for maintainer use.

## Evidence Quality

1. No evidence or fabricated evidence.
2. Weak evidence; misses obvious issue details.
3. Some correct evidence, but incomplete.
4. Good issue-grounded evidence.
5. Excellent evidence tied to symptoms, context, and likely impact.

## Calibration

1. Very overconfident or underconfident.
2. Confidence often mismatched.
3. Reasonable but inconsistent.
4. Well calibrated.
5. Explicitly handles uncertainty and missing information.

## Duplicate Judgment

1. Incorrect duplicate decision likely wastes maintainer time.
2. Weak duplicate comparison.
3. Plausible but not fully justified.
4. Correct duplicate/non-duplicate decision with evidence.
5. Finds the best duplicate cluster or explains why none applies.

## Prioritization Quality

1. Ignores severity, effort, or user impact.
2. Poor trade-offs.
3. Acceptable but shallow.
4. Good value/effort trade-off.
5. Maintainer-level sprint planning judgment.

## Recommended Human-Eval Protocol

Evaluate at least 100 held-out issues per repo. Each issue should be rated by two independent reviewers. Resolve disagreements larger than two points with a third reviewer. Report mean score, inter-rater agreement, and examples of severe failures.
