import argparse
import json
from pathlib import Path
from typing import Dict, List

from agents.baselines import BASELINES
from benchmark.audit_dataset import audit
from benchmark.dataset import deterministic_split, load_issues
from benchmark.evaluate import evaluate_task1, evaluate_task2, evaluate_task3
from benchmark.metrics import bootstrap_ci
from tasks.task1 import grade_task1
from tasks.task2 import grade_task2
from tasks.task3 import grade_task3


def task_reward_samples(agent, issues: List[Dict], task: str, batch_size: int = 5) -> List[float]:
    if task == "task1":
        return [grade_task1(agent.act_task1(issue), issue).score for issue in issues]
    if task == "task2":
        history = []
        rewards = []
        for issue in issues:
            rewards.append(grade_task2(agent.act_task2(issue, history), issue).score)
            history.append(issue)
        return rewards
    rewards = []
    for start in range(0, len(issues), batch_size):
        batch = issues[start:start + batch_size]
        if batch:
            rewards.append(grade_task3(agent.act_task3(batch), batch).score)
    return rewards


def ablation_report(data: List[Dict], baseline: str) -> Dict:
    variants = {
        "full": data,
        "no_comments": [{**issue, "comments": []} for issue in data],
        "no_labels": [{**issue, "labels": []} for issue in data],
        "title_only": [{**issue, "body": ""} for issue in data],
    }
    output = {}
    for name, variant in variants.items():
        train = deterministic_split(variant, "train")
        test = deterministic_split(variant, "test")
        agent = BASELINES[baseline]().fit(train)
        output[name] = {
            "task1": evaluate_task1(agent, test),
            "task2": evaluate_task2(agent, test),
            "task3": evaluate_task3(agent, test),
        }
    return output


def build_report(args):
    data = load_issues(args.data)
    train = deterministic_split(data, "train")
    test = deterministic_split(data, "test")
    results = {}
    intervals = {}

    for name in args.baselines:
        agent = BASELINES[name]().fit(train)
        results[name] = {
            "task1": evaluate_task1(agent, test),
            "task2": evaluate_task2(agent, test),
            "task3": evaluate_task3(agent, test),
        }
        intervals[name] = {
            task: bootstrap_ci(task_reward_samples(agent, test, task), samples=args.bootstrap_samples)
            for task in ["task1", "task2", "task3"]
        }

    payload = {
        "dataset_audit": audit(args.data),
        "baselines": args.baselines,
        "results": results,
        "confidence_intervals": intervals,
        "ablations": ablation_report(data, args.ablation_baseline),
        "human_eval": {
            "rubric": "docs/human_evaluation_rubric.md",
            "error_taxonomy": "docs/error_taxonomy.md",
            "template": "docs/benchmark_report_template.md",
        },
    }

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with Path(args.output).open("w") as f:
        json.dump(payload, f, indent=2)
    print(json.dumps(payload, indent=2))


def main():
    parser = argparse.ArgumentParser(description="Generate benchmark report with CIs and ablations.")
    parser.add_argument("--data", default="data/issues.json")
    parser.add_argument(
        "--baselines",
        nargs="+",
        default=["majority", "heuristic", "embedding-centroid", "hybrid-rl", "offline-rl"],
        choices=sorted(BASELINES),
    )
    parser.add_argument("--ablation-baseline", default="embedding-centroid", choices=sorted(BASELINES))
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--output", default="reports/benchmark_report.json")
    args = parser.parse_args()
    build_report(args)


if __name__ == "__main__":
    main()
