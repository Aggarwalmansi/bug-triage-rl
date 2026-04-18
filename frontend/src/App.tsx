import { useEffect, useMemo, useRef, useState } from "react";
import { API_BASE } from "./config";

type ContextIssue = {
  issue_id: number;
  title: string;
  repo: string;
  similarity?: number | null;
  severity?: string | null;
};

type Observation = {
  issue_id: number;
  title: string;
  body: string;
  repo: string;
  labels: string[];
  context: ContextIssue[];
  candidate_duplicates: ContextIssue[];
  metadata: Record<string, unknown>;
};

type ActionPayload = {
  issue_id: number;
  bug_type?: string;
  severity?: string;
  duplicate_of?: number | null;
  selected_issues?: number[] | null;
  component?: string;
  needs_info?: boolean;
  confidence?: number;
  rationale?: string[];
};

type DecisionTelemetry = {
  task: string;
  strategy: string;
  mode: string;
  epsilon: number;
  state_features: string[];
  q_values: Record<string, number>;
  action: ActionPayload;
};

type TrainingSummary = {
  episodes: number;
  start_avg: number;
  end_avg: number;
  best_avg: number;
  improvement: number;
};

type Metrics = {
  task: string;
  index: number;
  step_count: number;
  avg_reward: number;
  reward_history: number[];
  last_action: ActionPayload | null;
  last_decision: DecisionTelemetry | null;
  agent: string;
  configured_episodes: number;
  current_episode: number;
  completed_episodes: number;
  split: string;
  training_summary: TrainingSummary;
  epsilon: number;
  done: boolean;
};

type RewardPayload = {
  score: number;
  feedback?: string | null;
  breakdown: Record<string, number>;
};

type SessionResponse = {
  observation: Observation | null;
  metrics: Metrics;
};

type StepResponse = {
  action: ActionPayload;
  observation: Observation | null;
  reward: RewardPayload;
  done: boolean;
  info: {
    metrics: Metrics;
  };
};

type RunEvent = {
  id: string;
  issueId: number;
  title: string;
  strategy: string;
  reward: number;
  timestamp: string;
};

async function api<T>(path: string, init?: RequestInit): Promise<T> {
  const controller = new AbortController();
  const timeout = window.setTimeout(() => controller.abort(), 15000);
  let response: Response;
  try {
    response = await fetch(`${API_BASE}${path}`, {
      headers: {
        "Content-Type": "application/json",
        ...(init?.headers ?? {})
      },
      ...init,
      signal: controller.signal
    });
  } catch (error) {
    if (error instanceof DOMException && error.name === "AbortError") {
      throw new Error(`Timed out calling ${path}`);
    }
    throw error;
  } finally {
    window.clearTimeout(timeout);
  }

  if (!response.ok) {
    throw new Error(`${response.status} ${response.statusText}`);
  }
  return response.json() as Promise<T>;
}

function truncate(value: string, length: number) {
  return value.length <= length ? value : `${value.slice(0, length)}...`;
}

function humanize(value?: string | null) {
  return (value ?? "Not available").replaceAll("_", " ");
}

function normalizeDecision(value: Metrics["last_decision"]): DecisionTelemetry | null {
  if (!value || !Array.isArray((value as Partial<DecisionTelemetry>).state_features)) {
    return null;
  }
  return value;
}

function LineChart(props: { values: number[]; stroke: string; fill?: string; height?: number }) {
  const { values, stroke, fill = "transparent", height = 180 } = props;
  if (values.length === 0) {
    return <div className="empty-state">No live rewards yet.</div>;
  }

  const width = 620;
  const max = Math.max(...values, 1);
  const min = Math.min(...values, 0);
  const range = max - min || 1;
  const stepX = values.length === 1 ? width : width / (values.length - 1);
  const points = values
    .map((value, index) => {
      const x = index * stepX;
      const y = height - ((value - min) / range) * (height - 22) - 10;
      return `${x},${y}`;
    })
    .join(" ");
  const area = `0,${height} ${points} ${width},${height}`;

  return (
    <svg className="chart-svg" viewBox={`0 0 ${width} ${height}`} preserveAspectRatio="none">
      <polygon points={area} fill={fill} />
      <polyline fill="none" stroke={stroke} strokeWidth="3" points={points} strokeLinejoin="round" />
      {values.map((value, index) => {
        const x = index * stepX;
        const y = height - ((value - min) / range) * (height - 22) - 10;
        return <circle key={`${index}-${value}`} cx={x} cy={y} r="4" fill={stroke} />;
      })}
    </svg>
  );
}

export default function App() {
  const [task, setTask] = useState("task1");
  const [agent, setAgent] = useState("hybrid-rl");
  const [episodes, setEpisodes] = useState(8);
  const [busy, setBusy] = useState(false);
  const [autoRun, setAutoRun] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [observation, setObservation] = useState<Observation | null>(null);
  const [metrics, setMetrics] = useState<Metrics | null>(null);
  const [decision, setDecision] = useState<DecisionTelemetry | null>(null);
  const [lastAction, setLastAction] = useState<ActionPayload | null>(null);
  const [lastReward, setLastReward] = useState<RewardPayload | null>(null);
  const [events, setEvents] = useState<RunEvent[]>([]);
  const timerRef = useRef<number | null>(null);

  const rewardHistory = metrics?.reward_history ?? [];
  const qValues = useMemo(
    () => Object.entries(decision?.q_values ?? {}).sort((left, right) => Number(right[1]) - Number(left[1])),
    [decision]
  );
  const contextItems =
    task === "task3"
      ? (Array.isArray(observation?.metadata.sprint_batch) ? (observation?.metadata.sprint_batch as ContextIssue[]) : [])
      : observation?.candidate_duplicates?.length
        ? observation.candidate_duplicates
        : observation?.context ?? [];

  async function startSession() {
    setBusy(true);
    setAutoRun(false);
    setError(null);
    try {
      const response = await api<SessionResponse>("/agent/session", {
        method: "POST",
        body: JSON.stringify({
          task,
          agent,
          split: "all",
          episodes,
          use_groq: false,
          curriculum: false
        })
      });
      setObservation(response.observation);
      setMetrics(response.metrics);
      setDecision(normalizeDecision(response.metrics.last_decision));
      setLastAction(response.metrics.last_action);
      setLastReward(null);
      setEvents([]);
    } catch (requestError) {
      setError(requestError instanceof Error ? requestError.message : "Failed to start session");
    } finally {
      setBusy(false);
    }
  }

  async function runOneStep() {
    setBusy(true);
    setError(null);
    try {
      const response = await api<StepResponse>("/agent/step", { method: "POST" });
      const nextDecision = normalizeDecision(response.info.metrics.last_decision);
      setObservation(response.observation);
      setMetrics(response.info.metrics);
      setDecision(nextDecision);
      setLastAction(response.action);
      setLastReward(response.reward);
      setEvents((current) => [
        {
          id: `${Date.now()}-${response.action.issue_id}`,
          issueId: response.action.issue_id,
          title: observation?.title ?? `Issue #${response.action.issue_id}`,
          strategy: nextDecision?.strategy ?? "not_available",
          reward: response.reward.score,
          timestamp: new Date().toLocaleTimeString()
        },
        ...current
      ].slice(0, 8));
    } catch (requestError) {
      setError(requestError instanceof Error ? requestError.message : "Step failed");
      setAutoRun(false);
    } finally {
      setBusy(false);
    }
  }

  useEffect(() => {
    if (!autoRun || !observation || busy || metrics?.done) {
      if (timerRef.current) {
        window.clearTimeout(timerRef.current);
        timerRef.current = null;
      }
      return;
    }
    timerRef.current = window.setTimeout(() => {
      void runOneStep();
    }, 1000);
    return () => {
      if (timerRef.current) {
        window.clearTimeout(timerRef.current);
        timerRef.current = null;
      }
    };
  }, [autoRun, observation, busy, metrics?.done]);

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div>
          <div className="brand-pill">Issue Triage Studio</div>
          <h1 className="hero-title">Maintainer Console</h1>
          <p className="hero-copy">
            Real backend sessions, real policy output, real step rewards.
          </p>
        </div>

        <section className="panel control-panel">
          <div className="panel-header">
            <h2>Session</h2>
            <span className="status-dot" data-live={Boolean(observation) && !metrics?.done} />
          </div>

          <label className="field">
            <span>Task</span>
            <select value={task} onChange={(event) => setTask(event.target.value)}>
              <option value="task1">Classification</option>
              <option value="task2">Duplicate Detection</option>
              <option value="task3">Sprint Planning</option>
            </select>
          </label>

          <label className="field">
            <span>Agent</span>
            <select value={agent} onChange={(event) => setAgent(event.target.value)}>
              <option value="hybrid-rl">Hybrid RL</option>
              <option value="offline-rl">Offline RL</option>
              <option value="groq-triage">Groq Triage</option>
            </select>
          </label>

          <label className="field">
            <span>Episodes</span>
            <input
              type="number"
              min={1}
              max={100}
              step={1}
              value={episodes}
              onChange={(event) => setEpisodes(Number(event.target.value))}
            />
          </label>

          <div className="button-row">
            <button className="primary-button" onClick={() => void startSession()} disabled={busy}>
              Start Session
            </button>
            <button className="secondary-button" onClick={() => void runOneStep()} disabled={busy || !observation || metrics?.done}>
              Step
            </button>
            <button
              className="secondary-button"
              onClick={() => setAutoRun((value) => !value)}
              disabled={busy || !observation || metrics?.done}
            >
              {autoRun ? "Pause" : "Auto Run"}
            </button>
          </div>

          <div className="session-summary">
            <div>
              <span>Backend</span>
              <strong>{API_BASE || "same origin"}</strong>
            </div>
            <div>
              <span>Episodes</span>
              <strong>{metrics?.configured_episodes ?? episodes}</strong>
            </div>
            <div>
              <span>Progress</span>
              <strong>
                {metrics ? `${metrics.completed_episodes}/${metrics.configured_episodes}` : `0/${episodes}`}
              </strong>
            </div>
            <div>
              <span>Current Episode</span>
              <strong>{metrics?.current_episode ?? 1}</strong>
            </div>
            <div>
              <span>Total Steps</span>
              <strong>{metrics?.step_count ?? 0}</strong>
            </div>
            <div>
              <span>Avg Reward</span>
              <strong>{rewardHistory.length ? metrics?.avg_reward.toFixed(3) : "—"}</strong>
            </div>
          </div>

          {error ? <div className="error-banner">{error}</div> : null}
        </section>

        <section className="panel feed-panel">
          <div className="panel-header">
            <h2>Recent Steps</h2>
            <span className="panel-meta">{events.length} events</span>
          </div>
          <div className="feed-list">
            {events.length === 0 ? (
              <div className="empty-state">No steps yet.</div>
            ) : (
              events.map((item) => (
                <article key={item.id} className="feed-item">
                  <div>
                    <div className="feed-title">{truncate(item.title, 54)}</div>
                    <div className="feed-meta">
                      #{item.issueId} · {humanize(item.strategy)} · {item.timestamp}
                    </div>
                  </div>
                  <div className="reward-chip">{item.reward.toFixed(2)}</div>
                </article>
              ))
            )}
          </div>
        </section>
      </aside>

      <main className="main-grid">
        <section className="panel issue-panel">
          <div className="panel-header">
            <h2>Current Issue</h2>
            <span className="panel-meta">{observation ? `#${observation.issue_id}` : "no active issue"}</span>
          </div>
          {observation ? (
            <>
              <div className="issue-topline">
                <span className="repo-pill">{observation.repo}</span>
                {observation.labels.map((label) => (
                  <span key={label} className="label-pill">
                    {label}
                  </span>
                ))}
              </div>
              <h3 className="issue-title">{observation.title}</h3>
              <p className="issue-body">{truncate(observation.body || "No body provided.", 560)}</p>
            </>
          ) : (
            <div className="empty-state">Start a session to load a real issue.</div>
          )}
        </section>

        <section className="panel action-panel">
          <div className="panel-header">
            <h2>Decision</h2>
            <span className="panel-meta">current backend policy output</span>
          </div>
          {decision && lastAction ? (
            <>
              <div className="action-topline">
                <div className="strategy-chip">{humanize(decision.strategy)}</div>
                <div className="mode-chip">{humanize(decision.mode)}</div>
              </div>
              <div className="action-grid">
                <div className="action-field">
                  <span>Type</span>
                  <strong>{lastAction.bug_type ?? "—"}</strong>
                </div>
                <div className="action-field">
                  <span>Severity</span>
                  <strong>{lastAction.severity ?? "—"}</strong>
                </div>
                <div className="action-field">
                  <span>Component</span>
                  <strong>{lastAction.component ?? "—"}</strong>
                </div>
                <div className="action-field">
                  <span>Duplicate</span>
                  <strong>{String(lastAction.duplicate_of ?? "None")}</strong>
                </div>
                <div className="action-field">
                  <span>Selection</span>
                  <strong>{lastAction.selected_issues?.join(", ") || "None"}</strong>
                </div>
                <div className="action-field">
                  <span>Confidence</span>
                  <strong>{(lastAction.confidence ?? 0).toFixed(2)}</strong>
                </div>
              </div>
              {(lastAction.rationale ?? []).length > 0 ? (
                <div className="rationale-list">
                  {(lastAction.rationale ?? []).map((item) => (
                    <div key={item} className="rationale-item">
                      {item}
                    </div>
                  ))}
                </div>
              ) : null}
              {qValues.length > 0 ? (
                <div className="qvalue-list">
                  {qValues.map(([label, value]) => (
                    <div key={label} className="qvalue-row">
                      <span>{humanize(label)}</span>
                      <strong>{value.toFixed(3)}</strong>
                    </div>
                  ))}
                </div>
              ) : null}
            </>
          ) : (
            <div className="empty-state">No real decision yet.</div>
          )}
        </section>

        <section className="panel chart-panel">
          <div className="panel-header">
            <h2>Reward Trace</h2>
            <span className="panel-meta">real rewards from each backend step</span>
          </div>
          <LineChart values={rewardHistory} stroke="#6e8f93" fill="rgba(110, 143, 147, 0.16)" />
        </section>

        <section className="panel reward-panel">
          <div className="panel-header">
            <h2>Latest Reward</h2>
            <span className="panel-meta">current reward breakdown</span>
          </div>
          {lastReward ? (
            <>
              <div className="score-row">
                <strong className="reward-score">{lastReward.score.toFixed(3)}</strong>
                <span className="reward-feedback">{lastReward.feedback}</span>
              </div>
              <div className="breakdown-list">
                {Object.entries(lastReward.breakdown).map(([label, value]) => (
                  <div key={label} className="breakdown-item">
                    <span>{label}</span>
                    <strong>{value.toFixed(3)}</strong>
                  </div>
                ))}
              </div>
            </>
          ) : (
            <div className="empty-state">Take a step to see reward details.</div>
          )}
        </section>

        <section className="panel duplicate-panel">
          <div className="panel-header">
            <h2>Context</h2>
            <span className="panel-meta">duplicate candidates, history, or sprint batch</span>
          </div>
          {contextItems.length > 0 ? (
            <div className="candidate-list">
              {contextItems.map((item) => (
                <article key={item.issue_id} className="candidate-item">
                  <div>
                    <div className="candidate-title">{truncate(item.title, 60)}</div>
                    <div className="candidate-meta">
                      #{item.issue_id} · {item.repo}
                    </div>
                  </div>
                  <div className="candidate-score">
                    {typeof item.similarity === "number" ? item.similarity.toFixed(2) : item.severity ?? "—"}
                  </div>
                </article>
              ))}
            </div>
          ) : (
            <div className="empty-state">No context items for this issue yet.</div>
          )}
        </section>
      </main>
    </div>
  );
}
