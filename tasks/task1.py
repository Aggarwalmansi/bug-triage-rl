from models import Reward

def grade_task1(action, issue):
    type_score = 1.0 if action.bug_type == issue["type"] else 0.0
    severity_score = 1.0 if action.severity == issue["severity"] else 0.0
    needs_info_score = 1.0 if action.needs_info == issue.get("needs_info", False) else 0.0
    component_score = 0.0
    if action.component:
        component_score = 1.0 if action.component == issue.get("component") else 0.0

    total = (
        0.45 * type_score
        + 0.35 * severity_score
        + 0.10 * needs_info_score
        + 0.10 * component_score
    )

    return Reward(
        score=round(total, 4),
        feedback=(
            f"type: {type_score}, severity: {severity_score}, "
            f"needs_info: {needs_info_score}, component: {component_score}"
        ),
        breakdown={
            "type": type_score,
            "severity": severity_score,
            "needs_info": needs_info_score,
            "component": component_score,
        },
    )
