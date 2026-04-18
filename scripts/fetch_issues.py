import requests
import json

def detect_type(title, body, labels):
    text = (title + " " + body).lower()

    if "bug" in labels or "error" in text or "crash" in text:
        return "bug"
    elif "doc" in text or "readme" in text:
        return "docs"
    elif "how" in text or "?" in title:
        return "question"
    else:
        return "feature"


def detect_severity(title, body):
    text = (title + " " + body).lower()

    if "crash" in text or "failure" in text:
        return "P1"
    elif "error" in text:
        return "P2"
    elif "missing" in text or "slow" in text:
        return "P3"
    else:
        return "P4"

GITHUB_API = "https://api.github.com/repos"

repos = [
    ("microsoft", "vscode"),
    ("pallets", "flask"),
    ("numpy", "numpy")
]

issues_data = []
issue_id = 1

for owner, repo in repos:
    url = f"{GITHUB_API}/{owner}/{repo}/issues"

    params = {
        "state": "closed",
        "per_page": 50
    }

    response = requests.get(url, params=params)
    issues = response.json()

    for issue in issues:
        if "pull_request" in issue:
            continue

        labels = [l["name"] for l in issue.get("labels", [])]
        body = issue.get("body", "")

    # Remove templates
        body = body.split("<!--")[0]
        issue_obj = {
            "id": issue_id,
            "title": issue["title"],
            "body": issue.get("body", "")[:500],
            "repo": repo,
            "labels": labels,
            "type": detect_type(issue["title"], body, labels),
            "severity": detect_severity(issue["title"], body),
            "duplicate_of": None
        }

        # Remove templates

        # Filter bad issues
        if issue["title"].lower() in ["hi", "<spam>"]:
            continue

        if len(body.strip()) < 20:
            continue

        issues_data.append(issue_obj)
        issue_id += 1

with open("data/issues.json", "w") as f:
    json.dump(issues_data, f, indent=2)

print(f"Saved {len(issues_data)} issues")