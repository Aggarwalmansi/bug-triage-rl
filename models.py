from pydantic import BaseModel, Field
from typing import Dict, List, Optional


BUG_TYPES = ["bug", "feature", "question", "docs"]
SEVERITIES = ["P0", "P1", "P2", "P3", "P4"]


class ContextIssue(BaseModel):
    issue_id: int
    title: str
    repo: str
    labels: List[str] = Field(default_factory=list)
    similarity: Optional[float] = None
    duplicate_of: Optional[int] = None
    severity: Optional[str] = None
    effort: Optional[int] = None
    impact: Optional[int] = None


class Observation(BaseModel):
    issue_id: int
    title: str
    body: str
    repo: str
    labels: List[str] = Field(default_factory=list)
    context: List[ContextIssue] = Field(default_factory=list)
    candidate_duplicates: List[ContextIssue] = Field(default_factory=list)
    metadata: Dict[str, object] = Field(default_factory=dict)


class Action(BaseModel):
    issue_id: int
    bug_type: str = "bug"
    severity: str = "P2"
    duplicate_of: Optional[int] = None
    selected_issues: Optional[List[int]] = None
    component: Optional[str] = None
    owner: Optional[str] = None
    needs_info: bool = False
    confidence: Optional[float] = None
    rationale: List[str] = Field(default_factory=list)


class Reward(BaseModel):
    score: float
    feedback: Optional[str] = None
    breakdown: Dict[str, float] = Field(default_factory=dict)
