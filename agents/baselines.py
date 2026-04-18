from collections import Counter
from typing import Dict, List, Optional

from benchmark.dataset import duplicate_candidates, infer_component, needs_more_info
from models import Action
from tasks.task3 import compute_value
try:
    from agents.groq_agent import GroqTriageAgent
except Exception:
    GroqTriageAgent = None
from agents.embedding_baselines import EmbeddingCentroidBaseline
from agents.hybrid_rl_agent import ProductionHybridRLAgent
from agents.offline_rl_agent import OfflineTracePolicy


class MajorityBaseline:
    name = "majority"

    def __init__(self):
        self.bug_type = "bug"
        self.severity = "P3"

    def fit(self, issues: List[Dict]):
        if issues:
            self.bug_type = Counter(issue.get("type", "bug") for issue in issues).most_common(1)[0][0]
            self.severity = Counter(issue.get("severity", "P3") for issue in issues).most_common(1)[0][0]
        return self

    def act_task1(self, issue: Dict) -> Action:
        return Action(
            issue_id=issue["id"],
            bug_type=self.bug_type,
            severity=self.severity,
            component="unknown",
            needs_info=False,
            confidence=0.5,
            rationale=["majority label baseline"],
        )

    def act_task2(self, issue: Dict, history: List[Dict]) -> Action:
        return Action(issue_id=issue["id"], duplicate_of=None, confidence=0.5)

    def act_task3(self, issues: List[Dict], capacity: int = 15) -> Action:
        selected = []
        effort = 0
        for issue in issues:
            cost = issue.get("effort", 3)
            if effort + cost <= capacity:
                selected.append(issue["id"])
                effort += cost
        return Action(issue_id=issues[0]["id"], selected_issues=selected, confidence=0.5)


class HeuristicBaseline:
    name = "heuristic"

    def fit(self, issues: List[Dict]):
        return self

    def act_task1(self, issue: Dict) -> Action:
        text = f"{issue.get('title', '')} {issue.get('body', '')}".lower()
        labels = " ".join(issue.get("labels", [])).lower()
        if any(word in text or word in labels for word in ["docs", "documentation", "readme"]):
            bug_type = "docs"
        elif "?" in issue.get("title", "") or text.startswith("how "):
            bug_type = "question"
        elif any(word in text or word in labels for word in ["bug", "error", "crash", "fail", "exception"]):
            bug_type = "bug"
        else:
            bug_type = "feature"

        if any(word in text for word in ["security", "data loss", "crash", "failure", "regression"]):
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
            confidence=0.65,
            rationale=["keyword and label heuristic"],
        )

    def act_task2(self, issue: Dict, history: List[Dict]) -> Action:
        candidates = duplicate_candidates(issue, history, k=1)
        duplicate_of: Optional[int] = None
        confidence = 0.5
        if candidates and candidates[0].get("similarity", 0) >= 0.42:
            duplicate_of = candidates[0]["id"]
            confidence = candidates[0]["similarity"]
        return Action(
            issue_id=issue["id"],
            duplicate_of=duplicate_of,
            confidence=confidence,
            rationale=["lexical duplicate retrieval baseline"],
        )

    def act_task3(self, issues: List[Dict], capacity: int = 15) -> Action:
        ranked = sorted(
            issues,
            key=lambda issue: compute_value(issue) / max(1, issue.get("effort", 3)),
            reverse=True,
        )
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
            confidence=0.7,
            rationale=["value per effort greedy sprint baseline"],
        )


BASELINES = {
    MajorityBaseline.name: MajorityBaseline,
    HeuristicBaseline.name: HeuristicBaseline,
    EmbeddingCentroidBaseline.name: EmbeddingCentroidBaseline,
    ProductionHybridRLAgent.name: ProductionHybridRLAgent,
    OfflineTracePolicy.name: OfflineTracePolicy,
}

if GroqTriageAgent is not None:
    BASELINES[GroqTriageAgent.name] = GroqTriageAgent
