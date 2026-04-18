import math
from collections import Counter
from typing import Dict, Iterable, List, Sequence, Tuple

from benchmark.dataset import text_for_issue, tokenize


Vector = Dict[int, float]


def stable_hash(token: str, dims: int) -> int:
    value = 2166136261
    for byte in token.encode("utf-8"):
        value ^= byte
        value *= 16777619
        value &= 0xFFFFFFFF
    return value % dims


class HashingTextEmbedder:
    def __init__(self, dims: int = 2048):
        self.dims = dims
        self.idf: Dict[str, float] = {}

    def fit(self, texts: Iterable[str]):
        docs = [set(tokenize(text)) for text in texts]
        n = len(docs)
        df = Counter(token for doc in docs for token in doc)
        self.idf = {token: math.log((1 + n) / (1 + count)) + 1 for token, count in df.items()}
        return self

    def transform_one(self, text: str) -> Vector:
        counts = Counter(tokenize(text))
        vector: Vector = {}
        for token, count in counts.items():
            index = stable_hash(token, self.dims)
            sign = -1.0 if stable_hash(f"sign:{token}", 2) == 0 else 1.0
            vector[index] = vector.get(index, 0.0) + sign * count * self.idf.get(token, 1.0)
        return normalize(vector)

    def transform(self, texts: Iterable[str]) -> List[Vector]:
        return [self.transform_one(text) for text in texts]

    def issue_vector(self, issue: Dict) -> Vector:
        labels = " ".join(issue.get("labels", []))
        comments = " ".join(comment.get("body", "") for comment in issue.get("comments", [])[:5])
        return self.transform_one(f"{text_for_issue(issue)} {labels} {comments}")


def normalize(vector: Vector) -> Vector:
    norm = math.sqrt(sum(value * value for value in vector.values()))
    if norm == 0:
        return {}
    return {index: value / norm for index, value in vector.items()}


def cosine(left: Vector, right: Vector) -> float:
    if not left or not right:
        return 0.0
    if len(left) > len(right):
        left, right = right, left
    return sum(value * right.get(index, 0.0) for index, value in left.items())


def mean_vector(vectors: Sequence[Vector]) -> Vector:
    total: Vector = {}
    for vector in vectors:
        for index, value in vector.items():
            total[index] = total.get(index, 0.0) + value
    if vectors:
        total = {index: value / len(vectors) for index, value in total.items()}
    return normalize(total)


class EmbeddingIndex:
    def __init__(self, embedder: HashingTextEmbedder):
        self.embedder = embedder
        self.items: List[Tuple[Dict, Vector]] = []

    def fit(self, issues: Sequence[Dict]):
        self.embedder.fit(text_for_issue(issue) for issue in issues)
        self.items = [(issue, self.embedder.issue_vector(issue)) for issue in issues]
        return self

    def search(self, issue: Dict, k: int = 5) -> List[Dict]:
        query = self.embedder.issue_vector(issue)
        scored = []
        for candidate, vector in self.items:
            if candidate.get("id") == issue.get("id"):
                continue
            score = cosine(query, vector)
            if score > 0:
                item = dict(candidate)
                item["similarity"] = round(score, 4)
                scored.append(item)
        return sorted(scored, key=lambda item: item["similarity"], reverse=True)[:k]
