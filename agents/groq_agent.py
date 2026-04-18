from typing import Dict, List

from clients.groq_client import GroqJSONClient
from models import Action


class GroqTriageAgent:
    name = "groq-triage"

    def __init__(self, model: str = None):
        self.client = GroqJSONClient(model=model) if model else GroqJSONClient()

    def fit(self, issues: List[Dict]):
        return self

    def act_task1(self, issue: Dict) -> Action:
        prompt = f"""
You are a senior software-maintenance triage engineer.

Classify this issue using only JSON.

Allowed bug_type values: bug, feature, question, docs.
Allowed severity values: P0, P1, P2, P3, P4.
Use P0 only for active security/data-loss/release-blocking incidents.
Return component, whether more information is needed, confidence in [0,1],
and 1-3 short evidence-based rationale strings.

Issue:
Repo: {issue.get('repo')}
Title: {issue.get('title')}
Labels: {issue.get('labels')}
Body: {issue.get('body')}

JSON schema:
{{
  "bug_type": "bug",
  "severity": "P2",
  "component": "unknown",
  "needs_info": false,
  "confidence": 0.0,
  "rationale": ["..."]
}}
"""
        data = self.client.complete_json(
            prompt,
            {
                "bug_type": "bug",
                "severity": "P2",
                "component": "unknown",
                "needs_info": False,
                "confidence": 0.0,
                "rationale": ["Groq unavailable; fallback action used."],
            },
        )
        return Action(issue_id=issue["id"], **data)

    def act_task2(self, issue: Dict, history: List[Dict]) -> Action:
        candidates = [
            {"id": item["id"], "title": item["title"], "repo": item.get("repo"), "labels": item.get("labels", [])}
            for item in history[-10:]
        ]
        prompt = f"""
You are detecting duplicate software issues.

Return JSON only. duplicate_of must be one candidate id or null.
Prefer null unless the symptom, root cause, and affected behavior are substantially the same.

Current issue:
ID: {issue.get('id')}
Title: {issue.get('title')}
Body: {issue.get('body')}

Candidate previous issues:
{candidates}

JSON schema:
{{
  "duplicate_of": null,
  "confidence": 0.0,
  "rationale": ["..."]
}}
"""
        data = self.client.complete_json(
            prompt,
            {"duplicate_of": None, "confidence": 0.0, "rationale": ["Groq unavailable; fallback action used."]},
        )
        return Action(
            issue_id=issue["id"],
            duplicate_of=data.get("duplicate_of"),
            confidence=data.get("confidence"),
            rationale=data.get("rationale", []),
        )

    def act_task3(self, issues: List[Dict], capacity: int = 15) -> Action:
        batch = [
            {
                "id": issue["id"],
                "title": issue["title"],
                "severity": issue.get("severity"),
                "effort": issue.get("effort"),
                "impact": issue.get("impact"),
                "needs_info": issue.get("needs_info"),
            }
            for issue in issues
        ]
        prompt = f"""
You are planning a software sprint.

Select issues that maximize engineering value under capacity {capacity}.
Use JSON only. selected_issues must be ids from the batch.

Issues:
{batch}

JSON schema:
{{
  "selected_issues": [1, 2],
  "confidence": 0.0,
  "rationale": ["..."]
}}
"""
        data = self.client.complete_json(
            prompt,
            {"selected_issues": [], "confidence": 0.0, "rationale": ["Groq unavailable; fallback action used."]},
        )
        return Action(
            issue_id=issues[0]["id"],
            selected_issues=data.get("selected_issues", []),
            confidence=data.get("confidence"),
            rationale=data.get("rationale", []),
        )
