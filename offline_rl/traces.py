import argparse
import json
from pathlib import Path
from typing import Dict, List

from agents.baselines import BASELINES
from benchmark.dataset import deterministic_split, load_issues
from tasks.task1 import grade_task1
from tasks.task2 import grade_task2
from tasks.task3 import grade_task3


def issue_observation(issue: Dict) -> Dict:
    return {
        "issue_id": issue["id"],
        "title": issue.get("title", ""),
        "body": issue.get("body", ""),
        "repo": issue.get("repo", ""),
        "labels": issue.get("labels", []),
    }


def action_record(action) -> Dict:
    return {
        "bug_type": action.bug_type,
        "severity": action.severity,
        "duplicate_of": action.duplicate_of,
        "selected_issues": action.selected_issues,
        "component": action.component,
        "needs_info": action.needs_info,
        "confidence": action.confidence,
        "rationale": action.rationale,
    }


def build_traces(data: List[Dict], policy_name: str) -> List[Dict]:
    agent = BASELINES[policy_name]().fit(data)
    traces = []
    history = []

    for issue in data:
        action = agent.act_task1(issue)
        reward = grade_task1(action, issue)
        traces.append(
            {
                "task": "task1",
                "observation": issue_observation(issue),
                "action": action_record(action),
                "reward": reward.score,
                "feedback": reward.feedback,
            }
        )

        action = agent.act_task2(issue, history)
        reward = grade_task2(action, issue)
        traces.append(
            {
                "task": "task2",
                "observation": issue_observation(issue),
                "context_ids": [item["id"] for item in history[-10:]],
                "action": action_record(action),
                "reward": reward.score,
                "feedback": reward.feedback,
            }
        )
        history.append(issue)

    for start in range(0, len(data), 5):
        batch = data[start:start + 5]
        if not batch:
            continue
        action = agent.act_task3(batch)
        reward = grade_task3(action, batch)
        traces.append(
            {
                "task": "task3",
                "observation": {"batch": [issue_observation(issue) for issue in batch]},
                "action": action_record(action),
                "reward": reward.score,
                "feedback": reward.feedback,
            }
        )

    return traces


def main():
    parser = argparse.ArgumentParser(description="Build offline RL decision traces from a behavior policy.")
    parser.add_argument("--data", default="data/issues.json")
    parser.add_argument("--split", default="train", choices=["train", "validation", "test", "all"])
    parser.add_argument("--policy", default="heuristic", choices=sorted(BASELINES))
    parser.add_argument("--output", default="data/decision_traces.jsonl")
    args = parser.parse_args()

    data = load_issues(args.data)
    if args.split != "all":
        data = deterministic_split(data, args.split)
    traces = build_traces(data, args.policy)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with Path(args.output).open("w") as f:
        for trace in traces:
            f.write(json.dumps(trace) + "\n")
    print(f"saved={len(traces)} output={args.output}")


if __name__ == "__main__":
    main()
