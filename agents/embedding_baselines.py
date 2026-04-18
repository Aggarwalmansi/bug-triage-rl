from collections import defaultdict
from typing import Dict, List, Optional

from benchmark.embeddings import EmbeddingIndex, HashingTextEmbedder, cosine, mean_vector
from benchmark.dataset import text_for_issue
from models import Action
from tasks.task3 import compute_value


class EmbeddingCentroidBaseline:
    name = "embedding-centroid"

    def __init__(self, dims: int = 2048):
        self.embedder = HashingTextEmbedder(dims=dims)
        self.type_centroids = {}
        self.severity_centroids = {}
        self.component_centroids = {}
        self.index = EmbeddingIndex(self.embedder)

    def fit(self, issues: List[Dict]):
        self.embedder.fit(text_for_issue(issue) for issue in issues)
        vectors = [(issue, self.embedder.issue_vector(issue)) for issue in issues]
        self.type_centroids = self._centroids(vectors, "type")
        self.severity_centroids = self._centroids(vectors, "severity")
        self.component_centroids = self._centroids(vectors, "component")
        self.index.items = vectors
        return self

    def _centroids(self, vectors, key: str):
        groups = defaultdict(list)
        for issue, vector in vectors:
            value = issue.get(key)
            if value:
                groups[value].append(vector)
        return {label: mean_vector(items) for label, items in groups.items()}

    def _predict(self, issue: Dict, centroids: Dict[str, Dict[int, float]], fallback: str) -> str:
        vector = self.embedder.issue_vector(issue)
        if not centroids:
            return fallback
        return max(centroids, key=lambda label: cosine(vector, centroids[label]))

    def act_task1(self, issue: Dict) -> Action:
        return Action(
            issue_id=issue["id"],
            bug_type=self._predict(issue, self.type_centroids, "bug"),
            severity=self._predict(issue, self.severity_centroids, "P3"),
            component=self._predict(issue, self.component_centroids, "unknown"),
            needs_info=len(issue.get("body", "").strip()) < 120,
            confidence=0.6,
            rationale=["nearest embedding centroid baseline"],
        )

    def act_task2(self, issue: Dict, history: List[Dict]) -> Action:
        local_index = EmbeddingIndex(self.embedder)
        local_index.items = [(item, self.embedder.issue_vector(item)) for item in history]
        candidates = local_index.search(issue, k=1)
        duplicate_of: Optional[int] = None
        confidence = 0.0
        if candidates and candidates[0].get("similarity", 0.0) >= 0.55:
            duplicate_of = candidates[0]["id"]
            confidence = candidates[0]["similarity"]
        return Action(
            issue_id=issue["id"],
            duplicate_of=duplicate_of,
            confidence=confidence,
            rationale=["embedding nearest-neighbor duplicate baseline"],
        )

    def act_task3(self, issues: List[Dict], capacity: int = 15) -> Action:
        ranked = sorted(
            issues,
            key=lambda issue: (compute_value(issue), -issue.get("effort", 3)),
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
            confidence=0.65,
            rationale=["embedding-supervised labels plus value planning"],
        )
