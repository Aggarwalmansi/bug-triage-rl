import argparse
import json
import os
import re
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import requests


DUPLICATE_PATTERNS = [
    re.compile(r"duplicate of #(?P<num>\d+)", re.I),
    re.compile(r"duplicates #(?P<num>\d+)", re.I),
    re.compile(r"closed as duplicate of #(?P<num>\d+)", re.I),
    re.compile(r"dupe of #(?P<num>\d+)", re.I),
]


def github_headers() -> Dict[str, str]:
    headers = {
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    token = os.environ.get("GITHUB_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def get_json(url: str, params: Optional[Dict] = None, retries: int = 3):
    for attempt in range(retries):
        response = requests.get(url, headers=github_headers(), params=params, timeout=30)
        if response.status_code == 403 and response.headers.get("X-RateLimit-Remaining") == "0":
            reset = int(response.headers.get("X-RateLimit-Reset", "0"))
            sleep_for = max(1, reset - int(time.time()) + 2)
            time.sleep(min(sleep_for, 120))
            continue
        if response.status_code >= 500 and attempt < retries - 1:
            time.sleep(2 ** attempt)
            continue
        response.raise_for_status()
        return response.json()
    response.raise_for_status()


def paginated(url: str, params: Optional[Dict] = None, max_pages: Optional[int] = None) -> Iterable[Dict]:
    page = 1
    while True:
        query = dict(params or {})
        query.update({"page": page, "per_page": 100})
        items = get_json(url, query)
        if not items:
            break
        for item in items:
            yield item
        page += 1
        if max_pages and page > max_pages:
            break


def parse_duplicate_reference(text: str) -> Optional[int]:
    for pattern in DUPLICATE_PATTERNS:
        match = pattern.search(text or "")
        if match:
            return int(match.group("num"))
    return None


def infer_type(labels: List[str], title: str, body: str) -> str:
    text = f"{title} {body}".lower()
    label_text = " ".join(labels).lower()
    if any(word in label_text for word in ["documentation", "docs"]):
        return "docs"
    if "question" in label_text or title.strip().endswith("?"):
        return "question"
    if any(word in label_text for word in ["bug", "regression", "crash"]) or any(
        word in text for word in ["error", "crash", "exception", "traceback", "fails"]
    ):
        return "bug"
    return "feature"


def infer_severity(labels: List[str], title: str, body: str) -> str:
    text = f"{title} {body} {' '.join(labels)}".lower()
    if any(word in text for word in ["security", "data loss", "critical", "p0", "sev0"]):
        return "P0"
    if any(word in text for word in ["crash", "regression", "failure", "p1", "sev1"]):
        return "P1"
    if any(word in text for word in ["error", "broken", "unable", "p2", "sev2"]):
        return "P2"
    if any(word in text for word in ["slow", "minor", "p3", "sev3"]):
        return "P3"
    return "P4"


def normalize_github_issue(repo: str, issue: Dict, comments: List[Dict], events: List[Dict]) -> Dict:
    labels = [label["name"] for label in issue.get("labels", [])]
    body = issue.get("body") or ""
    duplicate_of = parse_duplicate_reference(body)

    for comment in comments:
        duplicate_of = duplicate_of or parse_duplicate_reference(comment.get("body") or "")
    for event in events:
        label = event.get("label") or {}
        if event.get("event") == "labeled" and "duplicate" in (label.get("name") or "").lower():
            duplicate_of = duplicate_of or parse_duplicate_reference(json.dumps(event))

    return {
        "id": int(issue["id"]),
        "github_number": int(issue["number"]),
        "repo": repo,
        "title": issue.get("title") or "",
        "body": body,
        "labels": labels,
        "state": issue.get("state"),
        "created_at": issue.get("created_at"),
        "closed_at": issue.get("closed_at"),
        "author_association": issue.get("author_association"),
        "comments": [
            {
                "id": comment.get("id"),
                "user": (comment.get("user") or {}).get("login"),
                "author_association": comment.get("author_association"),
                "created_at": comment.get("created_at"),
                "body": comment.get("body") or "",
            }
            for comment in comments
        ],
        "timeline_events": [
            {
                "event": event.get("event"),
                "created_at": event.get("created_at"),
                "actor": (event.get("actor") or {}).get("login"),
                "label": (event.get("label") or {}).get("name"),
                "assignee": (event.get("assignee") or {}).get("login"),
                "commit_id": event.get("commit_id"),
            }
            for event in events
        ],
        "type": infer_type(labels, issue.get("title") or "", body),
        "severity": infer_severity(labels, issue.get("title") or "", body),
        "duplicate_of": duplicate_of,
    }


def fetch_repo(owner: str, repo: str, target: int, include_timeline: bool) -> List[Dict]:
    full_repo = f"{owner}/{repo}"
    base = f"https://api.github.com/repos/{full_repo}"
    output = []
    for issue in paginated(f"{base}/issues", {"state": "all", "sort": "updated", "direction": "desc"}):
        if "pull_request" in issue:
            continue
        comments = list(paginated(issue["comments_url"], max_pages=2))
        events = []
        if include_timeline:
            timeline_url = f"{base}/issues/{issue['number']}/timeline"
            events = list(paginated(timeline_url, max_pages=2))
        output.append(normalize_github_issue(full_repo, issue, comments, events))
        if len(output) >= target:
            break
    return output


def main():
    parser = argparse.ArgumentParser(description="Build a large GitHub issue triage dataset.")
    parser.add_argument("--repos", nargs="+", default=["microsoft/vscode", "pallets/flask", "numpy/numpy"])
    parser.add_argument("--target-total", type=int, default=10000)
    parser.add_argument("--include-timeline", action="store_true")
    parser.add_argument("--output", default="data/github_issues_enriched.json")
    args = parser.parse_args()

    per_repo = max(1, args.target_total // len(args.repos))
    records = []
    for full_repo in args.repos:
        owner, repo = full_repo.split("/", 1)
        records.extend(fetch_repo(owner, repo, per_repo, args.include_timeline))
        if len(records) >= args.target_total:
            break

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with Path(args.output).open("w") as f:
        json.dump(records[: args.target_total], f, indent=2)
    print(f"saved={len(records[:args.target_total])} output={args.output}")


if __name__ == "__main__":
    main()
