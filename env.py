from models import Observation, Action, Reward
from benchmark.action_validation import sanitize_action
from benchmark.dataset import duplicate_candidates, normalize_issue, sample_curriculum

from tasks.task1 import grade_task1
from tasks.task2 import grade_task2
from tasks.task3 import grade_task3


class BugTriageEnv:
    def __init__(self, data, candidate_k=5, sprint_batch_size=5):
        self.data = [normalize_issue(issue) for issue in data]
        self.candidate_k = candidate_k
        self.sprint_batch_size = sprint_batch_size
        self.current_index = 0
        self.current_task = "task1"
        self.done = False
        self.history = []

    def reset(self, task="task1", curriculum=False):
        self.current_task = task
        self.current_index = 0
        self.done = False
        self.history = []
        if curriculum:
            self.data = sample_curriculum(self.data)

        return self._build_observation()

    def _build_observation(self):
        issue = self.data[self.current_index]
        candidates = duplicate_candidates(issue, self.history, self.candidate_k)
        metadata = {
            "severity": issue.get("severity"),
            "type": issue.get("type"),
            "effort": issue.get("effort"),
            "impact": issue.get("impact"),
            "component": issue.get("component"),
            "needs_info": issue.get("needs_info"),
        }
        if self.current_task == "task3":
            batch = self.data[self.current_index:self.current_index + self.sprint_batch_size]
            metadata["sprint_batch"] = [
                {
                    "issue_id": item["id"],
                    "title": item["title"],
                    "repo": item["repo"],
                    "severity": item.get("severity"),
                    "effort": item.get("effort"),
                    "impact": item.get("impact"),
                    "component": item.get("component"),
                }
                for item in batch
            ]
        return Observation(
            issue_id=issue["id"],
            title=issue["title"],
            body=issue["body"],
            repo=issue["repo"],
            labels=issue.get("labels", []),
            context=[
                {
                    "issue_id": item["id"],
                    "title": item["title"],
                    "repo": item["repo"],
                    "labels": item.get("labels", []),
                    "severity": item.get("severity"),
                    "effort": item.get("effort"),
                    "impact": item.get("impact"),
                    "duplicate_of": item.get("duplicate_of"),
                }
                for item in self.history[-10:]
            ],
            candidate_duplicates=[
                {
                    "issue_id": item["id"],
                    "title": item["title"],
                    "repo": item["repo"],
                    "labels": item.get("labels", []),
                    "similarity": item.get("similarity"),
                    "severity": item.get("severity"),
                    "effort": item.get("effort"),
                    "impact": item.get("impact"),
                    "duplicate_of": item.get("duplicate_of"),
                }
                for item in candidates
            ],
            metadata=metadata,
        )

    def step(self, action):
        issue = self.data[self.current_index]
        batch = []
        if self.current_task == "task3":
            batch = self.data[self.current_index:self.current_index + self.sprint_batch_size]
        action = sanitize_action(
            action,
            task=self.current_task,
            current_issue=issue,
            history=self.history,
            batch=batch,
        )

        if self.current_task == "task1":
            reward = grade_task1(action, issue)

        elif self.current_task == "task2":
            reward = grade_task2(action, issue)

        elif self.current_task == "task3":
            reward = grade_task3(action, batch)

        else:
            reward = Reward(score=0.0)

        self.history.append(issue)
        if self.current_task == "task3":
            self.current_index += self.sprint_batch_size
        else:
            self.current_index += 1

        if self.current_index >= len(self.data):
            self.done = True
            return None, reward, True, {}

        return self._build_observation(), reward, False, {}

    def state(self):
        try:
            return {
                "task": getattr(self, "current_task", None),
                "index": getattr(self, "current_index", 0),
                "done": getattr(self, "done", False)
            }
        except Exception as e:
            return {"error": str(e)}
