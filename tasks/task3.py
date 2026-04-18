from models import Reward

def compute_value(issue):
    severity_weight = {
        "P0": 5,
        "P1": 4,
        "P2": 3,
        "P3": 2,
        "P4": 1
    }
    severity_value = severity_weight.get(issue["severity"], 1)
    impact = issue.get("impact", severity_value)
    needs_info_penalty = 1 if issue.get("needs_info") else 0
    return max(1, severity_value + impact - needs_info_penalty)


def optimal_knapsack(issues, capacity):
    n = len(issues)
    dp = [[0 for _ in range(capacity + 1)] for _ in range(n + 1)]
    keep = [[False for _ in range(capacity + 1)] for _ in range(n + 1)]

    for i, issue in enumerate(issues, start=1):
        effort = issue.get("effort", 3)
        value = compute_value(issue)
        for cap in range(capacity + 1):
            dp[i][cap] = dp[i - 1][cap]
            if effort <= cap and dp[i - 1][cap - effort] + value > dp[i][cap]:
                dp[i][cap] = dp[i - 1][cap - effort] + value
                keep[i][cap] = True

    selected = []
    cap = capacity
    for i in range(n, 0, -1):
        if keep[i][cap]:
            issue = issues[i - 1]
            selected.append(issue["id"])
            cap -= issue.get("effort", 3)
    return dp[n][capacity], list(reversed(selected))


def grade_task3(action, issues, capacity=15):
    selected = action.selected_issues or []
    selected = list(dict.fromkeys(selected))

    total_effort = 0
    total_value = 0

    for issue in issues:
        if issue["id"] in selected:
            total_effort += issue.get("effort", 3)
            total_value += compute_value(issue)

    if total_effort > capacity:
        return Reward(
            score=0.0,
            feedback="Exceeded capacity",
            breakdown={"total_effort": total_effort, "capacity": capacity, "total_value": total_value},
        )

    best_value, best_selection = optimal_knapsack(issues, capacity)
    score = total_value / best_value if best_value > 0 else 0

    return Reward(
        score=round(score, 4),
        feedback=f"value: {total_value}/{best_value}; optimal: {best_selection}",
        breakdown={
            "total_value": total_value,
            "best_value": best_value,
            "total_effort": total_effort,
            "capacity": capacity,
        },
    )
