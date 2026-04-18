import os
from typing import Dict, List, Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from agents.baselines import BASELINES
from agents.hybrid_rl_agent import ProductionHybridRLAgent
from benchmark.action_validation import sanitize_action
from benchmark.dataset import deterministic_split, load_issues
from env import BugTriageEnv
from models import Action


def cors_allow_origins():
    defaults = ["http://localhost:5173", "http://127.0.0.1:5173"]
    configured = os.environ.get("CORS_ALLOW_ORIGINS", "")
    extra = [origin.strip() for origin in configured.split(",") if origin.strip()]
    return sorted(set(defaults + extra))


class AgentSessionRequest(BaseModel):
    task: str = "task1"
    agent: str = "hybrid-rl"
    split: str = "test"
    curriculum: bool = False
    episodes: int = 25
    use_groq: bool = False


class AgentTrainRequest(BaseModel):
    episodes: int = Field(default=25, ge=1, le=500)


class RuntimeState:
    def __init__(self, data: List[Dict]):
        self.data = data
        self.task = "task1"
        self.eval_split = "test"
        self.agent_name = ProductionHybridRLAgent.name
        self.use_groq = False
        self.episodes = 25
        self.reward_history: List[float] = []
        self.step_count = 0
        self.last_action = None
        self.last_decision: Dict = {}
        self.training_curve: List[Dict] = []
        self.training_summary: Dict = {}
        self.agent = None
        self.current_episode = 1
        self.completed_episodes = 0
        self.env = BugTriageEnv(self._eval_data())

    def _split_or_all(self, split: str) -> List[Dict]:
        if split == "all":
            return list(self.data)
        items = deterministic_split(self.data, split)
        return items if items else list(self.data)

    def _train_data(self) -> List[Dict]:
        items = self._split_or_all("train")
        return items if items else list(self.data)

    def _eval_data(self) -> List[Dict]:
        return self._split_or_all(self.eval_split)

    def _build_agent(self, name: str, episodes: Optional[int] = None, use_groq: Optional[bool] = None):
        if name == ProductionHybridRLAgent.name:
            return ProductionHybridRLAgent(
                episodes=episodes or self.episodes,
                use_groq=self.use_groq if use_groq is None else use_groq,
                verbose=False,
            )
        if name not in BASELINES:
            raise ValueError(f"Unknown agent: {name}")
        agent_factory = BASELINES[name]
        return agent_factory()

    def _refresh_agent_telemetry(self):
        self.training_curve = list(getattr(self.agent, "training_curve", []) or [])
        if hasattr(self.agent, "training_summary"):
            self.training_summary = self.agent.training_summary()
        else:
            self.training_summary = {
                "episodes": 0,
                "start_avg": 0.0,
                "end_avg": 0.0,
                "best_avg": 0.0,
                "improvement": 0.0,
            }
        self.last_decision = dict(getattr(self.agent, "last_decision", {}) or {})

    def configure_session(self, request: AgentSessionRequest):
        self.task = request.task
        self.eval_split = request.split
        self.agent_name = request.agent
        self.use_groq = request.use_groq
        self.episodes = request.episodes
        self.reward_history = []
        self.step_count = 0
        self.last_action = None
        self.last_decision = {}
        self.current_episode = 1
        self.completed_episodes = 0
        self.agent = self._build_agent(request.agent, request.episodes, request.use_groq)
        self.agent.fit(self._train_data())
        self._refresh_agent_telemetry()
        self.env = BugTriageEnv(self._eval_data())
        observation = self.env.reset(task=request.task, curriculum=request.curriculum)
        return observation

    def train_agent(self, request: AgentTrainRequest):
        if self.agent_name != ProductionHybridRLAgent.name:
            raise ValueError("Training is only supported for the hybrid RL agent.")
        self.episodes = request.episodes
        self.agent = self._build_agent(self.agent_name, request.episodes, self.use_groq)
        self.agent.fit(self._train_data())
        self._refresh_agent_telemetry()
        return self.training_summary

    def reset(self, task: Optional[str] = None, curriculum: bool = False):
        if task is not None:
            self.task = task
        self.reward_history = []
        self.step_count = 0
        self.last_action = None
        self.last_decision = {}
        self.current_episode = 1
        self.completed_episodes = 0
        return self.env.reset(task=self.task, curriculum=curriculum)

    def _current_issue(self) -> Dict:
        return self.env.data[self.env.current_index]

    def _current_batch(self) -> List[Dict]:
        if self.task != "task3":
            return []
        return self.env.data[self.env.current_index:self.env.current_index + self.env.sprint_batch_size]

    def _agent_action(self) -> Action:
        if self.agent is None:
            self.agent = self._build_agent(self.agent_name, self.episodes, self.use_groq)
            self.agent.fit(self._train_data())
            self._refresh_agent_telemetry()

        issue = self._current_issue()
        if self.task == "task1":
            action = self.agent.act_task1(issue)
            action = sanitize_action(action, "task1", issue)
        elif self.task == "task2":
            action = self.agent.act_task2(issue, self.env.history)
            action = sanitize_action(action, "task2", issue, history=self.env.history)
        else:
            batch = self._current_batch()
            action = self.agent.act_task3(batch)
            action = sanitize_action(action, "task3", batch[0], batch=batch)

        self.last_action = action.model_dump()
        decision = dict(getattr(self.agent, "last_decision", {}) or {})
        if not decision:
            decision = {
                "task": self.task,
                "strategy": self.agent_name,
                "mode": "policy",
                "epsilon": 0.0,
                "state_features": [],
                "q_values": {},
                "action": action.model_dump(),
            }
        self.last_decision = decision
        return action

    def record_step(self, action: Action, reward_score: float):
        self.last_action = action.model_dump()
        self.reward_history.append(float(reward_score))
        self.step_count += 1
        self.last_decision = dict(getattr(self.agent, "last_decision", {}) or self.last_decision)

    def run_agent_step(self):
        if self.completed_episodes >= self.episodes:
            return None, None, None, True, {}

        if self.env.done and self.completed_episodes < self.episodes:
            self.current_episode = self.completed_episodes + 1
            self.env.reset(task=self.task, curriculum=False)

        action = self._agent_action()
        observation, reward, done, info = self.env.step(action)
        self.record_step(action, reward.score)

        if done:
            self.completed_episodes += 1
            session_done = self.completed_episodes >= self.episodes
            if not session_done:
                self.current_episode = self.completed_episodes + 1
                observation = self.env.reset(task=self.task, curriculum=False)
                done = False
            else:
                self.current_episode = self.completed_episodes
                done = True

        return action, observation, reward, done, info

    def metrics(self, done: bool = False):
        avg_reward = sum(self.reward_history) / len(self.reward_history) if self.reward_history else 0.0
        telemetry = {}
        if hasattr(self.agent, "telemetry"):
            telemetry = self.agent.telemetry()
        return {
            "task": self.task,
            "index": self.env.current_index,
            "step_count": self.step_count,
            "avg_reward": round(avg_reward, 4),
            "reward_history": [round(value, 4) for value in self.reward_history],
            "last_action": self.last_action,
            "last_decision": self.last_decision,
            "agent": self.agent_name,
            "configured_episodes": self.episodes,
            "current_episode": self.current_episode,
            "completed_episodes": self.completed_episodes,
            "split": self.eval_split,
            "training_curve": self.training_curve,
            "training_summary": self.training_summary,
            "epsilon": telemetry.get("epsilon", 0.0),
            "q_table_sizes": telemetry.get("q_table_sizes", {}),
            "done": done,
        }

    def session_payload(self, observation):
        return {
            "observation": observation,
            "metrics": self.metrics(done=self.env.done),
        }


app = FastAPI(title="Bug Triage RL API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_allow_origins(),
    allow_origin_regex=r"https://.*\.hf\.space",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

data = load_issues("data/issues.json")
runtime = RuntimeState(data)


@app.get("/reset")
def reset(task: str = "task1", curriculum: bool = False):
    observation = runtime.reset(task=task, curriculum=curriculum)
    return observation


@app.post("/step")
def step(action: Action):
    current_issue = runtime.env.data[runtime.env.current_index]
    batch = runtime._current_batch()
    action = sanitize_action(
        action,
        task=runtime.task,
        current_issue=current_issue,
        history=runtime.env.history,
        batch=batch,
    )
    observation, reward, done, info = runtime.env.step(action)
    runtime.record_step(action, reward.score)
    return {
        "observation": observation,
        "reward": reward,
        "done": done,
        "info": {
            **info,
            "metrics": runtime.metrics(done=done),
        },
    }


@app.get("/state")
def state():
    return {
        **runtime.env.state(),
        **runtime.metrics(done=runtime.env.done),
    }


@app.post("/agent/session")
def agent_session(request: AgentSessionRequest):
    observation = runtime.configure_session(request)
    runtime._agent_action()
    return runtime.session_payload(observation)


@app.post("/agent/train")
def agent_train(request: AgentTrainRequest):
    summary = runtime.train_agent(request)
    return {
        "training_summary": summary,
        "training_curve": runtime.training_curve,
        "metrics": runtime.metrics(done=runtime.env.done),
    }


@app.post("/agent/act")
def agent_act():
    action = runtime._agent_action()
    return {
        "action": action,
        "decision": runtime.last_decision,
        "metrics": runtime.metrics(done=runtime.env.done),
    }


@app.post("/agent/step")
def agent_step():
    action, observation, reward, done, info = runtime.run_agent_step()
    if action is None:
        return {
            "action": None,
            "observation": None,
            "reward": {"score": 0.0, "feedback": "Session complete", "breakdown": {}},
            "done": True,
            "info": {
                **info,
                "metrics": runtime.metrics(done=True),
            },
        }
    return {
        "action": action,
        "observation": observation,
        "reward": reward,
        "done": done,
        "info": {
            **info,
            "metrics": runtime.metrics(done=done),
        },
    }


@app.get("/agent/state")
def agent_state():
    return runtime.metrics(done=runtime.env.done)
