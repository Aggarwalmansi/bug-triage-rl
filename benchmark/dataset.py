import json
import math
import random
import re
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Sequence


TOKEN_RE = re.compile(r"[a-zA-Z0-9_]+")


def tokenize(text: str) -> List[str]:
    return [t.lower() for t in TOKEN_RE.findall(text or "") if len(t) > 2]


def text_for_issue(issue: Dict) -> str:
    return f"{issue.get('title', '')} {issue.get('body', '')}"


def infer_effort(issue: Dict) -> int:
    text = text_for_issue(issue).lower()
    labels = " ".join(issue.get("labels", [])).lower()
    if any(w in text or w in labels for w in ["security", "regression", "crash", "failure"]):
        return 5
    if any(w in text or w in labels for w in ["api", "performance", "slow", "memory"]):
        return 4
    if any(w in text or w in labels for w in ["docs", "readme", "typo"]):
        return 1
    if issue.get("type") == "feature":
        return 4
    return 3


def infer_impact(issue: Dict) -> int:
    text = text_for_issue(issue).lower()
    labels = " ".join(issue.get("labels", [])).lower()
    severity = issue.get("severity", "P3")
    score = {"P0": 5, "P1": 5, "P2": 4, "P3": 2, "P4": 1}.get(severity, 2)
    if any(w in text or w in labels for w in ["crash", "data loss", "security", "vulnerability"]):
        score += 1
    if any(w in text or w in labels for w in ["minor", "docs", "typo"]):
        score -= 1
    return min(5, max(1, score))


def normalize_issue(issue: Dict) -> Dict:
    normalized = dict(issue)
    normalized.setdefault("labels", [])
    normalized.setdefault("type", normalized.get("bug_type", "bug"))
    normalized.setdefault("severity", "P3")
    normalized.setdefault("duplicate_of", None)
    normalized.setdefault("effort", infer_effort(normalized))
    normalized.setdefault("impact", infer_impact(normalized))
    normalized.setdefault("component", infer_component(normalized))
    normalized.setdefault("needs_info", needs_more_info(normalized))
    return normalized


def infer_component(issue: Dict) -> str:
    text = text_for_issue(issue).lower()
    labels = " ".join(issue.get("labels", [])).lower()
    component_rules = {
        "terminal": ["terminal", "shell", "console"],
        "auth": ["login", "sign in", "token", "tenant", "authentication", "authorization"],
        "api": ["api", "method", "route", "endpoint"],
        "docs": ["docs", "documentation", "readme"],
        "performance": ["slow", "performance", "memory", "cpu"],
        "ui": ["view", "highlight", "explorer", "render"],
    }
    joined = f"{text} {labels}"
    for component, words in component_rules.items():
        if any(w in joined for w in words):
            return component
    return "unknown"


def needs_more_info(issue: Dict) -> bool:
    body = issue.get("body", "") or ""
    title = issue.get("title", "") or ""
    return len(body.strip()) < 120 or title.strip().endswith("?")


def load_issues(path: str = "data/issues.json") -> List[Dict]:
    with Path(path).open() as f:
        data = json.load(f)
    return [normalize_issue(issue) for issue in data]


def deterministic_split(data: Sequence[Dict], split: str = "test") -> List[Dict]:
    buckets = {"train": [], "validation": [], "test": []}
    for issue in data:
        issue_id = int(issue["id"])
        if issue_id % 10 < 7:
            buckets["train"].append(issue)
        elif issue_id % 10 == 7:
            buckets["validation"].append(issue)
        else:
            buckets["test"].append(issue)
    return buckets[split]


def lexical_similarity(a: Dict, b: Dict) -> float:
    left = set(tokenize(text_for_issue(a)))
    right = set(tokenize(text_for_issue(b)))
    if not left or not right:
        return 0.0
    return len(left & right) / math.sqrt(len(left) * len(right))


def duplicate_candidates(issue: Dict, history: Iterable[Dict], k: int = 5) -> List[Dict]:
    ranked = []
    for candidate in history:
        if candidate["id"] == issue["id"]:
            continue
        score = lexical_similarity(issue, candidate)
        if score > 0:
            item = dict(candidate)
            item["similarity"] = round(score, 4)
            ranked.append(item)
    return sorted(ranked, key=lambda x: x["similarity"], reverse=True)[:k]


def label_distribution(data: Sequence[Dict], key: str) -> Counter:
    return Counter(issue.get(key) for issue in data)


def sample_curriculum(data: Sequence[Dict], seed: int = 7) -> List[Dict]:
    rng = random.Random(seed)
    easy = [x for x in data if x.get("needs_info") is False]
    hard = [x for x in data if x.get("needs_info") is True]
    rng.shuffle(easy)
    rng.shuffle(hard)
    return easy + hard
