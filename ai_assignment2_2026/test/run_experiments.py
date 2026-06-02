import gymnasium as gym
import numpy as np
import sys
import matplotlib.pyplot as plt
import os
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

# Create results directory if it doesn't exist
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

from agents.local_search_agents import (
    SimulatedAnnealingAgent,
    HillClimbingAgent,
    RandomizedHillClimbingAgent,
    TestingAgent
)
import ns_gym
from ns_gym.wrappers import NSClassicControlWrapper
from ns_gym.schedulers import ContinuousScheduler, PeriodicScheduler
from ns_gym.update_functions import RandomWalk, IncrementUpdate


def make_env(domain, timesteps, render=False):
    render_mode = "human" if render else "rgb_array"
    env = gym.make(domain, render_mode=render_mode, max_episode_steps=timesteps)
    name, version = domain.split("-")

    scheduler_1 = ContinuousScheduler()
    scheduler_2 = PeriodicScheduler(period=3)
    update_function1 = IncrementUpdate(scheduler_1, k=1)
    update_function2 = RandomWalk(scheduler_2)

    if name == "CartPole":
        tunable_params = {"masspole": update_function1, "gravity": update_function2}
    elif name == "MountainCar":
        tunable_params = {"gravity": update_function1, "force": update_function1}

    env = NSClassicControlWrapper(env, tunable_params, change_notification=True)
    return env, name


def make_agent(agent_name, num_actions, domain_name):
    if agent_name == "SimulatedAnnealing":
        return SimulatedAnnealingAgent(num_actions=num_actions, domain_name=domain_name)
    elif agent_name == "HillClimbing":
        return HillClimbingAgent(num_actions=num_actions, domain_name=domain_name)
    elif agent_name == "RandomizedHillClimbing":
        return RandomizedHillClimbingAgent(num_actions=num_actions, domain_name=domain_name)
    elif agent_name == "Random":
        return TestingAgent(num_actions=num_actions, domain_name=domain_name)


def run_episodes(domain, timesteps, n_episodes, seed=42):
    """
    Τρέχει n_episodes για κάθε agent, επιστρέφει
    dict: agent_name -> list of cumulative rewards
    """
    agent_names = ["Random", "HillClimbing", "RandomizedHillClimbing", "SimulatedAnnealing"]
    results = {name: [] for name in agent_names}

    for agent_name in agent_names:
        print(f"  Running {agent_name} on {domain} (timesteps={timesteps}, episodes={n_episodes})...")
        env, domain_name = make_env(domain, timesteps)

        for ep in range(n_episodes):
            agent = make_agent(agent_name, env.action_space.n, domain_name)
            _, rewards, _, _, _ = agent.find_best_route(env, seed=seed + ep)
            results[agent_name].append(np.sum(rewards))

        env.close()

    return results


# ──────────────────────────────────────────────
# EXPERIMENT 1:
# 100 episodes × max_timesteps in [10, 100, 500, 1000]
# → mean cumulative reward per (agent, timesteps)
# ──────────────────────────────────────────────
def experiment_1(domain):
    max_timesteps_list = [10, 100, 500, 1000]
    n_episodes = 100
    agent_names = ["Random", "HillClimbing", "RandomizedHillClimbing", "SimulatedAnnealing"]
    mean_rewards = {name: [] for name in agent_names}

    for ts in max_timesteps_list:
        print(f"\n[Exp1 | {domain}] max_timesteps={ts}")
        results = run_episodes(domain, ts, n_episodes)
        for name in agent_names:
            mean_rewards[name].append(np.mean(results[name]))

    x = np.arange(len(max_timesteps_list))
    width = 0.2
    fig, ax = plt.subplots(figsize=(12, 6))
    colors = ["#2196F3", "#FF9800", "#4CAF50", "#E91E63"]
    for i, (name, color) in enumerate(zip(agent_names, colors)):
        bars = ax.bar(x + i * width, mean_rewards[name], width,
                      label=name, color=color, alpha=0.85)
        ax.bar_label(bars, fmt="%.1f", padding=3, fontsize=9)

    ax.set_xlabel("Max Timesteps per Episode")
    ax.set_ylabel("Mean Cumulative Reward (100 episodes)")
    ax.set_title(f"Mean Cumulative Reward vs Max Timesteps — {domain}")
    ax.set_xticks(x + width * len(agent_names) / 2)
    ax.set_xticklabels([str(ts) for ts in max_timesteps_list])
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    fname = f"exp1_{domain.replace('-','_')}.png"
    plt.savefig(os.path.join(RESULTS_DIR, fname), dpi=150)
    plt.close()
    print(f"  Saved: {os.path.join(RESULTS_DIR, fname)}")
    return mean_rewards


# ──────────────────────────────────────────────
# EXPERIMENT 2:
# 1000 episodes × max_timesteps=100
# → cumulative reward per episode (learning curve)
# ──────────────────────────────────────────────
def experiment_2(domain):
    timesteps = 100
    n_episodes = 1000
    agent_names = ["Random", "HillClimbing", "RandomizedHillClimbing", "SimulatedAnnealing"]

    print(f"\n[Exp2 | {domain}] max_timesteps={timesteps}, episodes={n_episodes}")
    results = run_episodes(domain, timesteps, n_episodes)

    def smooth(data, window=50):
        w = min(window, len(data))
        return np.convolve(data, np.ones(w)/w, mode="valid")

    fig, ax = plt.subplots(figsize=(12, 6))
    colors = ["#2196F3", "#FF9800", "#4CAF50", "#E91E63"]
    for name, color in zip(agent_names, colors):
        raw = results[name]
        w = min(50, len(raw))
        ax.plot(raw, alpha=0.15, color=color)
        smoothed = smooth(raw, w)
        ax.plot(range(w - 1, len(raw)), smoothed,
                label=f"{name} (smooth)", color=color, linewidth=2)

    ax.set_xlabel("Episode")
    ax.set_ylabel("Cumulative Reward")
    ax.set_title(f"Cumulative Reward per Episode (max_timesteps=100) — {domain}")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    fname = f"exp2_{domain.replace('-','_')}.png"
    plt.savefig(os.path.join(RESULTS_DIR, fname), dpi=150)
    plt.close()
    print(f"  Saved: {os.path.join(RESULTS_DIR, fname)}")
    return results


if __name__ == "__main__":
    for domain in ["CartPole-v1", "MountainCar-v0"]:
        print(f"\n{'='*50}")
        print(f"DOMAIN: {domain}")
        print(f"{'='*50}")
        exp1 = experiment_1(domain)
        exp2 = experiment_2(domain)
