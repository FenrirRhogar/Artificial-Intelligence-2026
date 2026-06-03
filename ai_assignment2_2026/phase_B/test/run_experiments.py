'''
Experimental procedure for Phase 2 (heuristic search for decision processes).

For each domain (non-stationary CartPole and MountainCar) we evaluate two agents:
    1. MCTS + UCT              (standard, no mentor)
    2. MCTS + informed UCT     (UCT combined with prior knowledge from a DQN mentor)

The informed agent exploits the pretrained DQN policy through
agents.dqn.DQNAgent.get_guidance(state, noise_level=...). Three policy
reliability levels are produced by varying the noise level:

    noise_levels = {0, 0.1, 0.3}

For every (domain, algorithm, policy level) we run NUM_EPISODES episodes using the
DEFAULT number of timesteps and store the cumulative episodic reward. The whole
experiment is repeated NUM_REPEATS times. Results are plotted, one figure per domain.

NOTE: the standard MCTS+UCT agent does not use the mentor, so the policy level is
irrelevant for it; it is run once per domain as a baseline.

------------------------------------------------------------------------------------
Expected agent interface (to be implemented in agents/MCTS.py):

    class MCTSAgent(base.Agent):
        def __init__(self, env, **kwargs): ...
        def act(self, observation, env): ...

    class InformedMCTSAgent(base.Agent):
        def __init__(self, env, guidance_agent, noise_level=0.0, **kwargs): ...
        def act(self, observation, env): ...

If your class names / signatures differ, adjust make_agent() below.
------------------------------------------------------------------------------------
'''
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np
import gymnasium as gym
import ns_gym
from ns_gym.wrappers import NSClassicControlWrapper
from ns_gym.schedulers import ContinuousScheduler, PeriodicScheduler
from ns_gym.update_functions import RandomWalk, IncrementUpdate
from tqdm import tqdm

import cfgs.cfg_dqn as cfg
from agents.dqn import DQNAgent
# MCTS agents implemented in agents/MCTS.py
from agents.MCTS import MCTSAgent, InformedMCTSAgent

# ----------------------------- experiment configuration -----------------------------
DOMAINS = ['CartPole-v1', 'MountainCar-v0']
NOISE_LEVELS = [0.0, 0.1, 0.3]      # policy reliability levels for the informed agent
NUM_EPISODES = 500                   # episodes per (domain, algorithm, policy level)
NUM_REPEATS = 3                      # repetitions of the whole experiment
RENDER_MODE = None                   # 'human' only for debugging (slow); keep None

RESULTS_DIR = Path(__file__).resolve().parent / 'results'

# pretrained DQN mentor per domain (used by the informed UCT agent)
DQN_MODEL_PATHS = {
    'CartPole-v1':    'agents/DDQN_models/CartPole-v1/DDQN_episode_13271959.pth',
    'MountainCar-v0': 'agents/DDQN_models/MountainCar-v0/DDQN_episode_2118775.pth',
}


def make_ns_env(domain):
    '''Build the non-stationary environment exactly as in test/test_ns_agent.py.'''
    env = gym.make(domain, render_mode=RENDER_MODE)
    name, _ = domain.split("-")

    scheduler_1 = ContinuousScheduler()
    scheduler_2 = PeriodicScheduler(period=3)
    update_function1 = IncrementUpdate(scheduler_1, k=1)
    update_function2 = RandomWalk(scheduler_2)

    if name == "CartPole":
        tunable_params = {"masspole": update_function1, "gravity": update_function2}
    elif name == "MountainCar":
        tunable_params = {"gravity": update_function1, "force": update_function1}
    else:
        raise ValueError(f"unknown domain {domain}")

    ns_env = NSClassicControlWrapper(env, tunable_params, change_notification=True)
    return env, ns_env


def load_guidance_agent(domain, base_env):
    '''Load the pretrained DQN mentor for the given domain.'''
    params = dict(cfg.agent)
    params['state_size'] = base_env.observation_space.shape[0]
    params['action_size'] = base_env.action_space.n
    params['model_path'] = DQN_MODEL_PATHS[domain]
    return DQNAgent(**params)


def make_agent(algorithm, planning_env, guidance_agent=None, noise_level=0.0):
    '''Factory for the MCTS agents. Adjust here if your constructors differ.'''
    if algorithm == 'uct':
        return MCTSAgent(planning_env)
    elif algorithm == 'informed_uct':
        return InformedMCTSAgent(planning_env, guidance_agent, noise_level=noise_level)
    raise ValueError(f"unknown algorithm {algorithm}")


def run_config(domain, algorithm, noise_level):
    '''Run NUM_EPISODES episodes for one configuration, return list of episodic rewards.'''
    base_env, ns_env = make_ns_env(domain)
    guidance_agent = load_guidance_agent(domain, base_env) if algorithm == 'informed_uct' else None

    rewards = []
    for _ in tqdm(range(NUM_EPISODES),
                  desc=f"{domain} | {algorithm} | noise={noise_level}", leave=False):
        obs, info = ns_env.reset()
        planning_env = ns_env.get_planning_env()
        agent = make_agent(algorithm, planning_env, guidance_agent, noise_level)

        episode_reward = 0
        done = truncated = False
        while not (done or truncated):
            action = agent.act(obs, planning_env)
            obs, reward, done, truncated, info = ns_env.step(action)
            if RENDER_MODE == 'human':
                base_env.render()
            planning_env = ns_env.get_planning_env()
            episode_reward += reward
        rewards.append(episode_reward)

    base_env.close()
    return rewards


def experiment_labels():
    '''(label, algorithm, noise_level) for every curve we plot.'''
    labels = [("MCTS+UCT", 'uct', None)]
    for nl in NOISE_LEVELS:
        labels.append((f"MCTS+informed UCT (noise={nl})", 'informed_uct', nl))
    return labels


def run_all():
    RESULTS_DIR.mkdir(exist_ok=True)
    # results[domain][label] = array of shape (NUM_REPEATS, NUM_EPISODES)
    results = {}
    for domain in DOMAINS:
        results[domain] = {}
        for label, algorithm, noise_level in experiment_labels():
            repeats = []
            for r in range(NUM_REPEATS):
                print(f"[{domain}] {label} -- repeat {r + 1}/{NUM_REPEATS}")
                repeats.append(run_config(domain, algorithm, noise_level or 0.0))
            arr = np.asarray(repeats, dtype=np.float32)
            results[domain][label] = arr
            np.save(RESULTS_DIR / f"{domain}__{label}.npy", arr)
    return results


def plot_results(results):
    import matplotlib.pyplot as plt
    for domain in DOMAINS:
        plt.figure(figsize=(10, 6))
        for label, _, _ in experiment_labels():
            arr = results[domain][label]           # (NUM_REPEATS, NUM_EPISODES)
            mean = arr.mean(axis=0)
            std = arr.std(axis=0)
            x = np.arange(1, mean.shape[0] + 1)
            plt.plot(x, mean, label=label)
            plt.fill_between(x, mean - std, mean + std, alpha=0.2)
        plt.title(f"Cumulative episodic reward -- {domain}")
        plt.xlabel("Episode")
        plt.ylabel("Cumulative reward")
        plt.legend()
        plt.grid(True, alpha=0.3)
        out = RESULTS_DIR / f"{domain}_rewards.png"
        plt.savefig(out, dpi=150, bbox_inches='tight')
        print(f"saved plot: {out}")
        plt.close()


if __name__ == '__main__':
    results = run_all()
    plot_results(results)
