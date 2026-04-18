from collections import Counter, defaultdict
from typing import Dict, List, Optional, Tuple

from agents.groq_agent import GroqTriageAgent
from benchmark.action_validation import sanitize_action
from benchmark.dataset import duplicate_candidates, infer_component, needs_more_info, tokenize
from models import Action
from tasks.task1 import grade_task1
from tasks.task2 import grade_task2
from tasks.task3 import compute_value, grade_task3


SEVERITY_RANK = {"P0": 5, "P1": 4, "P2": 3, "P3": 2, "P4": 1}


class ProductionHybridRLAgent:
    name = "hybrid-rl"

    def __init__(
        self,
        episodes: int = 25,
        alpha: float = 0.25,
        gamma: float = 0.2,
        epsilon: float = 0.35,
        epsilon_decay: float = 0.9,
        min_epsilon: float = 0.03,
        use_groq: bool = False,
        verbose: bool = False,
    ):
        self.episodes = episodes
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.use_groq = use_groq
        self.verbose = verbose
        self.llm = GroqTriageAgent() if use_groq else None
        self.q = {
            "task1": defaultdict(lambda: defaultdict(float)),
            "task2": defaultdict(lambda: defaultdict(float)),
            "task3": defaultdict(lambda: defaultdict(float)),
        }
        self.task_actions = {
            "task1": ["safe_heuristic", "metadata_adjusted", "llm_adjusted"],
            "task2": ["safe_none", "candidate_threshold_low", "candidate_threshold_high", "llm_validated"],
            "task3": ["value_effort", "severity_impact_frequency", "top3_safe"],
        }
        self.training_curve: List[Dict] = []
        self.last_decision: Dict = {}

    def fit(self, issues: List[Dict]):
        if not issues:
            return self

        for episode in range(1, self.episodes + 1):
            totals = {"task1": [], "task2": [], "task3": []}
            history: List[Dict] = []

            for issue in issues:
                totals["task1"].append(self._train_task1(issue))
                totals["task2"].append(self._train_task2(issue, history))
                history.append(issue)

            for start in range(0, len(issues), 5):
                batch = issues[start:start + 5]
                if batch:
                    totals["task3"].append(self._train_task3(batch, history=issues[:start]))

            summary = {
                "episode": episode,
                "epsilon": round(self.epsilon, 4),
                "task1": self._mean(totals["task1"]),
                "task2": self._mean(totals["task2"]),
                "task3": self._mean(totals["task3"]),
            }
            summary["avg"] = round((summary["task1"] + summary["task2"] + summary["task3"]) / 3, 4)
            self.training_curve.append(summary)
            if self.verbose:
                print(
                    "Episode "
                    f"{episode:02d} | avg={summary['avg']:.3f} "
                    f"task1={summary['task1']:.3f} task2={summary['task2']:.3f} "
                    f"task3={summary['task3']:.3f} epsilon={summary['epsilon']:.3f}"
                )
            self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

        if self.verbose and self.training_curve:
            first = self.training_curve[0]["avg"]
            last = self.training_curve[-1]["avg"]
            print(f"Improvement: {first:.3f} -> {last:.3f} ({last - first:+.3f})")
        return self

    def _train_task1(self, issue: Dict) -> float:
        state = self.extract_state("task1", issue)
        action_name = self.choose_strategy("task1", state)
        action = self._task1_action(issue, action_name)
        reward = grade_task1(action, issue).score
        self.learn("task1", state, action_name, reward)
        for candidate in self.task_actions["task1"]:
            if candidate != action_name:
                candidate_reward = grade_task1(self._task1_action(issue, candidate), issue).score
                self.learn("task1", state, candidate, candidate_reward)
        return reward

    def _train_task2(self, issue: Dict, history: List[Dict]) -> float:
        state = self.extract_state("task2", issue, history)
        action_name = self.choose_strategy("task2", state)
        action = self._task2_action(issue, history, action_name)
        reward = grade_task2(action, issue).score
        self.learn("task2", state, action_name, reward)
        for candidate in self.task_actions["task2"]:
            if candidate != action_name:
                candidate_reward = grade_task2(self._task2_action(issue, history, candidate), issue).score
                self.learn("task2", state, candidate, candidate_reward)
        return reward

    def _train_task3(self, batch: List[Dict], history: List[Dict]) -> float:
        state = self.extract_state("task3", batch[0], history, batch)
        action_name = self.choose_strategy("task3", state)
        action = self._task3_action(batch, action_name, history)
        reward = grade_task3(action, batch).score
        self.learn("task3", state, action_name, reward)
        for candidate in self.task_actions["task3"]:
            if candidate != action_name:
                candidate_reward = grade_task3(self._task3_action(batch, candidate, history), batch).score
                self.learn("task3", state, candidate, candidate_reward)
        return reward

    def extract_state(
        self,
        task: str,
        issue: Dict,
        history: Optional[List[Dict]] = None,
        batch: Optional[List[Dict]] = None,
    ) -> str:
        history = history or []
        batch = batch or []
        text = f"{issue.get('title', '')} {issue.get('body', '')}".lower()
        heuristic = self.safe_task1_heuristic(issue)
        metadata = {
            "type": heuristic.bug_type,
            "severity": heuristic.severity,
            "component": heuristic.component,
            "impact": issue.get("impact", 0),
            "effort": issue.get("effort", 0),
            "needs_info": heuristic.needs_info,
        }
        features = [
            f"type:{metadata['type']}",
            f"sev:{metadata['severity']}",
            f"comp:{metadata['component']}",
            f"impact:{self._bucket(metadata['impact'])}",
            f"effort:{self._bucket(metadata['effort'])}",
            f"needs_info:{metadata['needs_info']}",
            f"err:{any(w in text for w in ['crash', 'error', 'fail', 'exception', 'traceback'])}",
            f"feature:{any(w in text for w in ['feature', 'add', 'support', 'request'])}",
        ]
        if task == "task2":
            candidates = duplicate_candidates(issue, history, k=3)
            top_similarity = candidates[0].get("similarity", 0.0) if candidates else 0.0
            features.extend([f"history:{self._bucket(len(history))}", f"dup_sim:{self._sim_bucket(top_similarity)}"])
        elif task == "task3":
            components = Counter(item.get("component") or infer_component(item) for item in batch)
            max_frequency = max(components.values()) if components else 0
            avg_value = self._mean(compute_value(item) for item in batch)
            features.extend([f"batch:{len(batch)}", f"freq:{self._bucket(max_frequency)}", f"value:{self._bucket(avg_value)}"])
        return f"{task}|" + "|".join(features)

    def choose_strategy(self, task: str, state: str) -> str:
        import random

        actions = self.task_actions[task]
        if random.random() < self.epsilon:
            return random.choice(actions)
        values = self.q[task][state]
        return max(actions, key=lambda action: values[action])

    def learn(self, task: str, state: str, action_name: str, reward: float):
        current = self.q[task][state][action_name]
        self.q[task][state][action_name] = current + self.alpha * (reward - current)

    def act_task1(self, issue: Dict) -> Action:
        state = self.extract_state("task1", issue)
        strategy = self.best_strategy("task1", state)
        action = self._task1_action(issue, strategy)
        self._record_decision("task1", state, strategy, action, issue=issue)
        return action

    def act_task2(self, issue: Dict, history: List[Dict]) -> Action:
        state = self.extract_state("task2", issue, history)
        strategy = self.best_strategy("task2", state)
        action = self._task2_action(issue, history, strategy)
        self._record_decision("task2", state, strategy, action, issue=issue, history=history)
        return action

    def act_task3(self, issues: List[Dict], capacity: int = 15) -> Action:
        state = self.extract_state("task3", issues[0], batch=issues)
        strategy = self.best_strategy("task3", state)
        action = self._task3_action(issues, strategy, capacity=capacity)
        self._record_decision("task3", state, strategy, action, issue=issues[0], batch=issues)
        return action

    def best_strategy(self, task: str, state: str) -> str:
        values = self.q[task][state]
        actions = self.task_actions[task]
        if not values or all(values[action] == 0.0 for action in actions):
            return "metadata_adjusted" if task == "task1" else "safe_none" if task == "task2" else "severity_impact_frequency"
        return max(actions, key=lambda action: values[action])

    def _task1_action(self, issue: Dict, strategy: str) -> Action:
        heuristic = self.safe_task1_heuristic(issue)
        if strategy == "metadata_adjusted":
            action = Action(
                issue_id=issue["id"],
                bug_type=heuristic.bug_type,
                severity=heuristic.severity,
                component=heuristic.component,
                needs_info=heuristic.needs_info,
                confidence=0.85,
                rationale=["metadata-adjusted RL decision"],
            )
        elif strategy == "llm_adjusted" and self.llm is not None:
            action = self.llm.act_task1(issue)
            if action.component in [None, "unknown"]:
                action.component = heuristic.component
            if action.bug_type not in ["bug", "feature", "question", "docs"]:
                action.bug_type = heuristic.bug_type
            if action.severity not in ["P0", "P1", "P2", "P3", "P4"]:
                action.severity = heuristic.severity
            action.rationale = (action.rationale or []) + ["RL selected LLM-adjusted policy"]
        else:
            action = heuristic
        return sanitize_action(action, "task1", issue)

    def _task2_action(self, issue: Dict, history: List[Dict], strategy: str) -> Action:
        threshold = 0.55 if strategy == "candidate_threshold_high" else 0.35
        candidates = duplicate_candidates(issue, history, k=1)
        duplicate_of = None
        confidence = 0.7
        rationale = ["safe no-duplicate fallback"]

        if strategy in ["candidate_threshold_low", "candidate_threshold_high"] and candidates:
            if candidates[0].get("similarity", 0.0) >= threshold:
                duplicate_of = candidates[0]["id"]
                confidence = candidates[0]["similarity"]
                rationale = [f"candidate similarity >= {threshold}"]
        elif strategy == "llm_validated" and self.llm is not None:
            llm_action = self.llm.act_task2(issue, history)
            duplicate_of = llm_action.duplicate_of
            confidence = llm_action.confidence or 0.5
            rationale = (llm_action.rationale or []) + ["RL selected LLM-validated duplicate policy"]

        return sanitize_action(
            Action(issue_id=issue["id"], duplicate_of=duplicate_of, confidence=confidence, rationale=rationale),
            "task2",
            issue,
            history=history,
        )

    def _task3_action(
        self,
        issues: List[Dict],
        strategy: str,
        history: Optional[List[Dict]] = None,
        capacity: int = 15,
    ) -> Action:
        history = history or []
        if strategy == "top3_safe":
            ranked = sorted(issues, key=lambda item: self._priority_score(item, history), reverse=True)[:3]
        elif strategy == "value_effort":
            ranked = sorted(
                issues,
                key=lambda item: self._priority_score(item, history) / max(1, item.get("effort", 3)),
                reverse=True,
            )
        else:
            ranked = sorted(issues, key=lambda item: self._priority_score(item, history), reverse=True)

        selected = []
        effort = 0
        for issue in ranked:
            cost = issue.get("effort", 3)
            if effort + cost <= capacity:
                selected.append(issue["id"])
                effort += cost
            if len(selected) >= 3:
                break

        return sanitize_action(
            Action(
                issue_id=issues[0]["id"],
                selected_issues=selected,
                confidence=0.8,
                rationale=["RL sprint policy using severity, impact, effort, and frequency"],
            ),
            "task3",
            issues[0],
            batch=issues,
        )

    def safe_task1_heuristic(self, issue: Dict) -> Action:
        text = f"{issue.get('title', '')} {issue.get('body', '')}".lower()
        labels = " ".join(issue.get("labels", [])).lower()
        if any(word in text or word in labels for word in ["docs", "documentation", "readme"]):
            bug_type = "docs"
        elif issue.get("title", "").strip().endswith("?") or text.startswith("how "):
            bug_type = "question"
        elif any(word in text or word in labels for word in ["bug", "crash", "error", "exception", "fail"]):
            bug_type = "bug"
        else:
            bug_type = "feature"

        if any(word in text for word in ["security", "data loss", "critical"]):
            severity = "P0"
        elif any(word in text for word in ["crash", "failure", "regression"]):
            severity = "P1"
        elif any(word in text for word in ["error", "broken", "unable"]):
            severity = "P2"
        elif any(word in text for word in ["slow", "missing", "inconsistent"]):
            severity = "P3"
        else:
            severity = "P4"

        return Action(
            issue_id=issue["id"],
            bug_type=bug_type,
            severity=severity,
            component=infer_component(issue),
            needs_info=needs_more_info(issue),
            confidence=0.7,
            rationale=["safe heuristic fallback"],
        )

    def _priority_score(self, issue: Dict, history: List[Dict]) -> float:
        heuristic = self.safe_task1_heuristic(issue)
        severity = SEVERITY_RANK.get(heuristic.severity, 2)
        impact = float(issue.get("impact", severity))
        effort = max(1, float(issue.get("effort", 3)))
        component = heuristic.component
        
        history_components = [self.safe_task1_heuristic(item).component for item in history]
        component_frequency = history_components.count(component)
        
        frequency_bonus = min(3, component_frequency) * 0.5
        needs_info_penalty = 1.0 if heuristic.needs_info else 0.0
        return severity * 2.0 + impact * 1.5 + frequency_bonus - needs_info_penalty - 0.2 * effort

    def print_learning_summary(self):
        if not self.training_curve:
            print("No RL training curve available.")
            return
        for row in self.training_curve:
            print(
                f"Episode {row['episode']:02d} | avg={row['avg']:.3f} "
                f"task1={row['task1']:.3f} task2={row['task2']:.3f} "
                f"task3={row['task3']:.3f} epsilon={row['epsilon']:.3f}"
            )
        first = self.training_curve[0]["avg"]
        last = self.training_curve[-1]["avg"]
        print(f"Improvement: {first:.3f} -> {last:.3f} ({last - first:+.3f})")

    def telemetry(self) -> Dict:
        return {
            "agent": self.name,
            "epsilon": round(self.epsilon, 4),
            "training_curve": self.training_curve,
            "training_summary": self.training_summary(),
            "last_decision": self.last_decision,
            "q_table_sizes": {task: len(values) for task, values in self.q.items()},
        }

    def training_summary(self) -> Dict:
        if not self.training_curve:
            return {
                "episodes": 0,
                "start_avg": 0.0,
                "end_avg": 0.0,
                "best_avg": 0.0,
                "improvement": 0.0,
            }
        start = self.training_curve[0]["avg"]
        end = self.training_curve[-1]["avg"]
        best = max(row["avg"] for row in self.training_curve)
        return {
            "episodes": len(self.training_curve),
            "start_avg": round(start, 4),
            "end_avg": round(end, 4),
            "best_avg": round(best, 4),
            "improvement": round(end - start, 4),
        }

    def _record_decision(
        self,
        task: str,
        state: str,
        strategy: str,
        action: Action,
        issue: Optional[Dict] = None,
        history: Optional[List[Dict]] = None,
        batch: Optional[List[Dict]] = None,
    ):
        self.last_decision = {
            "task": task,
            "strategy": strategy,
            "mode": "exploit",
            "epsilon": round(self.epsilon, 4),
            "state": state,
            "state_features": state.split("|")[1:],
            "q_values": self._q_values(task, state),
            "action": action.model_dump(),
            "issue_id": issue.get("id") if issue else None,
            "history_size": len(history or []),
            "batch_size": len(batch or []),
        }

    def _q_values(self, task: str, state: str) -> Dict[str, float]:
        values = self.q[task][state]
        return {action: round(values[action], 4) for action in self.task_actions[task]}

    def _bucket(self, value) -> str:
        try:
            value = float(value)
        except (TypeError, ValueError):
            return "unknown"
        if value <= 1:
            return "low"
        if value <= 3:
            return "mid"
        return "high"

    def _sim_bucket(self, value: float) -> str:
        if value >= 0.55:
            return "high"
        if value >= 0.35:
            return "mid"
        return "low"

    def _mean(self, values) -> float:
        values = list(values)
        return round(sum(values) / len(values), 4) if values else 0.0
