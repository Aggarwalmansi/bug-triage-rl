from typing import Dict, Iterable, List
import random


def mean(values: Iterable[float]) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def classification_metrics(rows: List[Dict], prediction_key: str, target_key: str) -> Dict[str, float]:
    total = len(rows)
    correct = sum(1 for row in rows if row[prediction_key] == row[target_key])
    return {"accuracy": correct / total if total else 0.0}


def duplicate_metrics(rows: List[Dict]) -> Dict[str, float]:
    tp = fp = tn = fn = 0
    for row in rows:
        pred = row.get("pred_duplicate_of")
        target = row.get("duplicate_of")
        if pred is None and target is None:
            tn += 1
        elif pred is not None and target is None:
            fp += 1
        elif pred is None and target is not None:
            fn += 1
        elif pred == target:
            tp += 1
        else:
            fp += 1
            fn += 1
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    accuracy = (tp + tn) / (tp + tn + fp + fn) if tp + tn + fp + fn else 0.0
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "positive_rate": (tp + fn) / (tp + tn + fp + fn) if tp + tn + fp + fn else 0.0,
    }


def sprint_metrics(rows: List[Dict]) -> Dict[str, float]:
    return {
        "mean_reward": mean(row.get("reward", 0.0) for row in rows),
        "capacity_violations": sum(1 for row in rows if row.get("capacity_violation")) / len(rows)
        if rows
        else 0.0,
    }


def bootstrap_ci(values: List[float], samples: int = 1000, confidence: float = 0.95, seed: int = 7) -> Dict[str, float]:
    if not values:
        return {"mean": 0.0, "low": 0.0, "high": 0.0}
    rng = random.Random(seed)
    means = []
    for _ in range(samples):
        draw = [rng.choice(values) for _ in values]
        means.append(mean(draw))
    means.sort()
    alpha = 1.0 - confidence
    low_index = int((alpha / 2) * (len(means) - 1))
    high_index = int((1 - alpha / 2) * (len(means) - 1))
    return {"mean": mean(values), "low": means[low_index], "high": means[high_index]}
