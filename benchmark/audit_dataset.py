import argparse
import json
from collections import Counter
from pathlib import Path

from benchmark.dataset import load_issues


def audit(path: str):
    data = load_issues(path)
    duplicate_positive = sum(1 for issue in data if issue.get("duplicate_of") is not None)
    effort_present = sum(1 for issue in data if "effort" in issue)
    impact_present = sum(1 for issue in data if "impact" in issue)
    return {
        "dataset": path,
        "num_issues": len(data),
        "repos": Counter(issue.get("repo") for issue in data),
        "types": Counter(issue.get("type") for issue in data),
        "severities": Counter(issue.get("severity") for issue in data),
        "duplicate_positive": duplicate_positive,
        "duplicate_positive_rate": duplicate_positive / len(data) if data else 0.0,
        "effort_present_after_normalization": effort_present,
        "impact_present_after_normalization": impact_present,
        "warnings": [
            warning
            for warning in [
                "Dataset is too small for research-grade claims." if len(data) < 1000 else "",
                "No positive duplicate labels; duplicate accuracy is misleading." if duplicate_positive == 0 else "",
                "Labels appear seed-scale; validate against human or maintainer decisions before claiming learning."
                if len(data) < 100
                else "",
            ]
            if warning
        ],
    }


def main():
    parser = argparse.ArgumentParser(description="Audit dataset readiness for benchmark claims.")
    parser.add_argument("--data", default="data/issues.json")
    parser.add_argument("--output", default="reports/dataset_audit.json")
    args = parser.parse_args()

    result = audit(args.data)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with Path(args.output).open("w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
