import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

from benchmark.dataset import infer_component, text_for_issue, tokenize
from models import Action
from tasks.task3 import compute_value


def state_key(issue: Dict) -> str:
    tokens = set(tokenize(text_for_issue(issue)))
    flags = [
        "crash" if {"crash", "failure", "exception"} & tokens else "no-crash",
        "docs" if {"docs", "documentation", "readme"} & tokens else "no-docs",
        "question" if issue.get("title", "").strip().endswith("?") else "statement",
        infer_component(issue),
    ]
    return "|".join(flags)


class OfflineTracePolicy:
    name = "offline-rl"

    def __init__(self, trace_path: str = "data/decision_traces.jsonl"):
        self.trace_path = trace_path
        self.task1_values = defaultdict(lambda: defaultdict(list))
        self.task2_values = defaultdict(lambda: defaultdict(list))

    def fit(self, issues: List[Dict]):
        path = Path(self.trace_path)
        if not path.exists():
            return self
        with path.open() as f:
            for line in f:
                if not line.strip():
                    continue
                trace = json.loads(line)
                obs = trace.get("observation", {})
                action = trace.get("action", {})
                reward = float(trace.get("reward", 0.0))
                if trace.get("task") == "task1":
                    key = state_key(obs)
                    action_key = f"{action.get('bug_type', 'bug')}|{action.get('severity', 'P3')}|{action.get('component') or 'unknown'}|{bool(action.get('needs_info'))}"
                    self.task1_values[key][action_key].append(reward)
                elif trace.get("task") == "task2":
                    key = state_key(obs)
                    action_key = "duplicate" if action.get("duplicate_of") is not None else "none"
                    self.task2_values[key][action_key].append(reward)
        return self

    def _best(self, table, key: str):
        values = table.get(key) or {}
        if not values:
            return None
        return max(values, key=lambda action: sum(values[action]) / len(values[action]))

    def act_task1(self, issue: Dict) -> Action:
        best = self._best(self.task1_values, state_key(issue))
        if not best:
            return Action(
                issue_id=issue["id"],
                bug_type="bug",
                severity="P3",
                component=infer_component(issue),
                needs_info=len(issue.get("body", "").strip()) < 120,
                confidence=0.35,
                rationale=["offline trace policy fallback"],
            )
        bug_type, severity, component, needs_info = best.split("|")
        return Action(
            issue_id=issue["id"],
            bug_type=bug_type,
            severity=severity,
            component=component,
            needs_info=needs_info == "True",
            confidence=0.7,
            rationale=["best historical offline trace action for similar state"],
        )

    def act_task2(self, issue: Dict, history: List[Dict]) -> Action:
        best = self._best(self.task2_values, state_key(issue))
        return Action(
            issue_id=issue["id"],
            duplicate_of=None,
            confidence=0.6 if best == "none" else 0.3,
            rationale=["offline trace duplicate policy"],
        )

    def act_task3(self, issues: List[Dict], capacity: int = 15) -> Action:
        ranked = sorted(issues, key=lambda issue: compute_value(issue), reverse=True)
        selected = []
        effort = 0
        for issue in ranked:
            cost = issue.get("effort", 3)
            if effort + cost <= capacity:
                selected.append(issue["id"])
                effort += cost
        return Action(
            issue_id=issues[0]["id"],
            selected_issues=selected,
            confidence=0.55,
            rationale=["offline policy value fallback for sprint planning"],
        )
