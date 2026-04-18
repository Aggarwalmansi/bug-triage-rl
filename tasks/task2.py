from models import Reward

def grade_task2(action, issue):
    true_dup = issue.get("duplicate_of")
    predicted_dup = action.duplicate_of

    if true_dup is None and predicted_dup is None:
        score = 1.0
        false_positive_penalty = 0.0
    elif predicted_dup == true_dup:
        score = 1.0
        false_positive_penalty = 0.0
    else:
        false_positive_penalty = 0.25 if true_dup is None and predicted_dup is not None else 0.0
        score = max(0.0, 0.2 - false_positive_penalty)

    return Reward(
        score=round(score, 4),
        feedback=f"expected: {true_dup}, got: {predicted_dup}",
        breakdown={
            "exact_match": 1.0 if predicted_dup == true_dup else 0.0,
            "false_positive_penalty": false_positive_penalty,
        },
    )
