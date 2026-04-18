import argparse
import json

from benchmark.dataset import load_issues
from benchmark.embeddings import EmbeddingIndex, HashingTextEmbedder


def main():
    parser = argparse.ArgumentParser(description="Search similar issues with the benchmark embedding index.")
    parser.add_argument("--data", default="data/issues.json")
    parser.add_argument("--issue-id", type=int, required=True)
    parser.add_argument("--k", type=int, default=5)
    args = parser.parse_args()

    data = load_issues(args.data)
    by_id = {issue["id"]: issue for issue in data}
    if args.issue_id not in by_id:
        raise SystemExit(f"issue_id={args.issue_id} not found")

    index = EmbeddingIndex(HashingTextEmbedder()).fit(data)
    results = index.search(by_id[args.issue_id], k=args.k)
    print(
        json.dumps(
            [
                {
                    "id": issue["id"],
                    "repo": issue.get("repo"),
                    "title": issue.get("title"),
                    "similarity": issue.get("similarity"),
                    "duplicate_of": issue.get("duplicate_of"),
                }
                for issue in results
            ],
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
