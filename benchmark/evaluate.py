import argparse
import json
from pathlib import Path
from typing import Dict, List

from agents.baselines import BASELINES
from agents.hybrid_rl_agent import ProductionHybridRLAgent
from benchmark.action_validation import sanitize_action
from benchmark.dataset import deterministic_split, load_issues
from benchmark.metrics import classification_metrics, duplicate_metrics, mean, sprint_metrics
from tasks.task1 import grade_task1
from tasks.task2 import grade_task2
from tasks.task3 import grade_task3


def evaluate_task1(agent, issues: List[Dict]) -> Dict:
    rows = []
    rewards = []
    for issue in issues:
        action = sanitize_action(agent.act_task1(issue), "task1", issue)
        reward = grade_task1(action, issue)
        rewards.append(reward.score)
        rows.append(
            {
                "issue_id": issue["id"],
                "pred_type": action.bug_type,
                "type": issue["type"],
                "pred_severity": action.severity,
                "severity": issue["severity"],
                "reward": reward.score,
            }
        )
    return {
        "reward": mean(rewards),
        "type": classification_metrics(rows, "pred_type", "type"),
        "severity": classification_metrics(rows, "pred_severity", "severity"),
    }


def evaluate_task2(agent, issues: List[Dict]) -> Dict:
    rows = []
    rewards = []
    history = []
    for issue in issues:
        action = sanitize_action(agent.act_task2(issue, history), "task2", issue, history=history)
        reward = grade_task2(action, issue)
        rewards.append(reward.score)
        rows.append(
            {
                "issue_id": issue["id"],
                "pred_duplicate_of": action.duplicate_of,
                "duplicate_of": issue.get("duplicate_of"),
                "reward": reward.score,
            }
        )
        history.append(issue)
    metrics = duplicate_metrics(rows)
    metrics["reward"] = mean(rewards)
    metrics["warning"] = (
        "No positive duplicate labels in this split; accuracy can be misleading."
        if metrics["positive_rate"] == 0
        else ""
    )
    return metrics


def evaluate_task3(agent, issues: List[Dict], batch_size: int = 5, capacity: int = 15) -> Dict:
    rows = []
    for start in range(0, len(issues), batch_size):
        batch = issues[start:start + batch_size]
        if not batch:
            continue
        action = sanitize_action(agent.act_task3(batch, capacity=capacity), "task3", batch[0], batch=batch)
        reward = grade_task3(action, batch, capacity=capacity)
        rows.append(
            {
                "batch_start": start,
                "selected_issues": action.selected_issues,
                "reward": reward.score,
                "capacity_violation": reward.feedback == "Exceeded capacity",
                "feedback": reward.feedback,
            }
        )
    return sprint_metrics(rows)


def run(args):
    data = load_issues(args.data)
    train = deterministic_split(data, "train")
    evaluation = deterministic_split(data, args.split)
    results = {}

    for name in args.baselines:
        if name == ProductionHybridRLAgent.name:
            agent = ProductionHybridRLAgent(episodes=args.hybrid_episodes).fit(train)
        else:
            agent = BASELINES[name]().fit(train)
        results[name] = {
            "task1": evaluate_task1(agent, evaluation),
            "task2": evaluate_task2(agent, evaluation),
            "task3": evaluate_task3(agent, evaluation, args.batch_size, args.capacity),
        }

    payload = {
        "dataset": args.data,
        "split": args.split,
        "num_train": len(train),
        "num_eval": len(evaluation),
        "results": results,
    }

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with Path(args.output).open("w") as f:
            json.dump(payload, f, indent=2)
    print(json.dumps(payload, indent=2))


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate bug-triage benchmark baselines.")
    parser.add_argument("--data", default="data/issues.json")
    parser.add_argument("--split", default="test", choices=["train", "validation", "test"])
    parser.add_argument("--baselines", nargs="+", default=["majority", "heuristic"], choices=sorted(BASELINES))
    parser.add_argument("--batch-size", type=int, default=5)
    parser.add_argument("--capacity", type=int, default=15)
    parser.add_argument("--output", default="reports/baseline_results.json")
    parser.add_argument("--hybrid-episodes", type=int, default=25)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
