import argparse

from agents.baselines import BASELINES
from agents.hybrid_rl_agent import ProductionHybridRLAgent
from benchmark.dataset import load_issues
from env import BugTriageEnv


def run_episode(env, agent, task):
    obs = env.reset(task=task)
    rewards = []
    history = []

    while True:
        issue = env.data[env.current_index]
        if task == "task1":
            action = agent.act_task1(issue)
        elif task == "task2":
            action = agent.act_task2(issue, history)
        else:
            batch = env.data[env.current_index:env.current_index + env.sprint_batch_size]
            action = agent.act_task3(batch)

        obs, reward, done, _ = env.step(action)
        rewards.append(reward.score)
        history.append(issue)
        if done:
            break

    return sum(rewards) / len(rewards) if rewards else 0.0


def main():
    parser = argparse.ArgumentParser(description="Run a triage agent through the environment.")
    parser.add_argument("--data", default="data/issues.json")
    parser.add_argument("--agent", default="heuristic", choices=sorted(BASELINES))
    parser.add_argument("--task", default="task1", choices=["task1", "task2", "task3"])
    parser.add_argument("--episodes", type=int, default=25)
    parser.add_argument("--use-groq", action="store_true")
    parser.add_argument("--show-learning", action="store_true")
    args = parser.parse_args()

    data = load_issues(args.data)
    env = BugTriageEnv(data)
    if args.agent == ProductionHybridRLAgent.name:
        agent = ProductionHybridRLAgent(
            episodes=args.episodes,
            use_groq=args.use_groq,
            verbose=args.show_learning,
        ).fit(data)
    else:
        agent = BASELINES[args.agent]().fit(data)
    score = run_episode(env, agent, args.task)
    print(f"agent={args.agent} task={args.task} score={score:.4f}")


if __name__ == "__main__":
    main()
