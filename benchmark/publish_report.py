import argparse
import json
from pathlib import Path


def fmt_ci(ci):
    return f"{ci.get('mean', 0):.3f} [{ci.get('low', 0):.3f}, {ci.get('high', 0):.3f}]"


def render(report):
    lines = [
        "# Bug Triage RL Benchmark Report",
        "",
        "## Dataset Audit",
        "",
    ]
    audit = report["dataset_audit"]
    lines.extend(
        [
            f"- Issues: {audit['num_issues']}",
            f"- Duplicate positive rate: {audit['duplicate_positive_rate']:.3f}",
            f"- Repositories: {dict(audit['repos'])}",
            "",
            "### Warnings",
            "",
        ]
    )
    for warning in audit.get("warnings", []):
        lines.append(f"- {warning}")

    lines.extend(["", "## Results", ""])
    for name, tasks in report["results"].items():
        lines.append(f"### {name}")
        for task, metrics in tasks.items():
            ci = report["confidence_intervals"].get(name, {}).get(task, {})
            reward = metrics.get("reward", metrics.get("mean_reward", 0.0))
            lines.append(f"- {task}: reward={reward:.3f}, bootstrap={fmt_ci(ci)}")
        lines.append("")

    lines.extend(["## Ablations", ""])
    for variant, tasks in report.get("ablations", {}).items():
        task1 = tasks.get("task1", {})
        task3 = tasks.get("task3", {})
        lines.append(
            f"- {variant}: task1_reward={task1.get('reward', 0):.3f}, "
            f"task3_reward={task3.get('mean_reward', 0):.3f}"
        )

    lines.extend(
        [
            "",
            "## Human Evaluation",
            "",
            f"- Rubric: {report['human_eval']['rubric']}",
            f"- Error taxonomy: {report['human_eval']['error_taxonomy']}",
            f"- Template: {report['human_eval']['template']}",
            "",
            "## Interpretation",
            "",
            "This report is valid as a pipeline smoke test on the seed dataset. "
            "It is not sufficient for research-grade claims until the enriched GitHub dataset "
            "contains thousands of issues, positive duplicate clusters, and human or maintainer-validated labels.",
        ]
    )
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description="Render benchmark report JSON as Markdown.")
    parser.add_argument("--input", default="reports/benchmark_report.json")
    parser.add_argument("--output", default="reports/benchmark_report.md")
    args = parser.parse_args()

    with Path(args.input).open() as f:
        report = json.load(f)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(render(report))
    print(f"saved={args.output}")


if __name__ == "__main__":
    main()
