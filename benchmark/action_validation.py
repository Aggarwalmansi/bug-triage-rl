from typing import Dict, Iterable, List, Optional

from models import Action, BUG_TYPES, SEVERITIES


def clamp_confidence(value):
    try:
        if value is None:
            return None
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return None


def valid_issue_ids(items: Iterable[Dict]) -> set:
    ids = set()
    for item in items:
        if "id" in item:
            ids.add(int(item["id"]))
        elif "issue_id" in item:
            ids.add(int(item["issue_id"]))
    return ids


def sanitize_duplicate_id(value, valid_ids: set) -> Optional[int]:
    if value in [None, "", "null"]:
        return None
    try:
        duplicate_id = int(value)
    except (TypeError, ValueError):
        return None
    if duplicate_id == 0:
        return None
    return duplicate_id if duplicate_id in valid_ids else None


def sanitize_selected_ids(values, valid_ids: set, limit: Optional[int] = None) -> List[int]:
    if not isinstance(values, list):
        return []
    selected = []
    for value in values:
        try:
            issue_id = int(value)
        except (TypeError, ValueError):
            continue
        if issue_id == 0 or issue_id not in valid_ids or issue_id in selected:
            continue
        selected.append(issue_id)
        if limit and len(selected) >= limit:
            break
    return selected


def sanitize_action(
    action: Action,
    task: str,
    current_issue: Dict,
    history: Optional[List[Dict]] = None,
    batch: Optional[List[Dict]] = None,
) -> Action:
    history = history or []
    batch = batch or []

    bug_type = action.bug_type if action.bug_type in BUG_TYPES else "bug"
    severity = action.severity if action.severity in SEVERITIES else "P2"
    duplicate_of = action.duplicate_of
    selected_issues = action.selected_issues

    if task == "task2":
        duplicate_of = sanitize_duplicate_id(action.duplicate_of, valid_issue_ids(history))
    else:
        duplicate_of = None

    if task == "task3":
        selected_issues = sanitize_selected_ids(action.selected_issues, valid_issue_ids(batch))
    else:
        selected_issues = None

    return Action(
        issue_id=int(current_issue["id"]),
        bug_type=bug_type,
        severity=severity,
        duplicate_of=duplicate_of,
        selected_issues=selected_issues,
        component=action.component or "unknown",
        owner=action.owner,
        needs_info=bool(action.needs_info),
        confidence=clamp_confidence(action.confidence),
        rationale=[str(item) for item in (action.rationale or [])][:3],
    )
