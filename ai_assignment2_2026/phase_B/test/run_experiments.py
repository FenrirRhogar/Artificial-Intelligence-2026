'''
Full experiments for Phase 2 (heuristic search for decision processes).

Domains : CartPole-v1, MountainCar-v0
Episodes: 500 per (domain, algorithm, noise_level)
Repeats : 3
Noise   : {0, 0.1, 0.3}  -- policy reliability levels for InformedMCTS

Usage:
    python run_experiments.py                            # all algorithms, all domains
    python run_experiments.py --algorithm mcts           # plain MCTS only
    python run_experiments.py --algorithm inf_mcts       # informed MCTS only
    python run_experiments.py --domain cartpole          # CartPole only
    python run_experiments.py -a mcts -d mountaincar
'''
import sys
import argparse
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
from agents.MCTS import MCTSAgent, InformedMCTSAgent

# ---------------------------------------------------------------------------
DOMAINS      = ['CartPole-v1', 'MountainCar-v0']
DOMAIN_MAP   = {
    'cartpole':    'CartPole-v1',
    'mountaincar': 'MountainCar-v0',
}
NOISE_LEVELS = [0.0, 0.1, 0.3]
NUM_EPISODES = 2
NUM_REPEATS  = 1
RENDER_MODE  = None   # overridden by --render flag

RESULTS_DIR = Path(__file__).resolve().parent / 'results'

PHASE_B_DIR = Path(__file__).resolve().parents[1]
DQN_MODEL_PATHS = {
    'CartPole-v1':    str(PHASE_B_DIR / 'agents/DDQN_models/CartPole-v1/DDQN_episode_13271959.pth'),
    'MountainCar-v0': str(PHASE_B_DIR / 'agents/DDQN_models/MountainCar-v0/DDQN_episode_2118775.pth'),
}
# ---------------------------------------------------------------------------


def make_ns_env(domain):
    env = gym.make(domain, render_mode=RENDER_MODE)
    name, _ = domain.split('-')

    scheduler_1 = ContinuousScheduler()
    scheduler_2 = PeriodicScheduler(period=3)

    # NOTE: IncrementUpdate adds k *per fire*. Defaults are tiny
    # (MountainCar force=0.001, gravity=0.0025; CartPole masspole=0.1),
    # so k must be scaled per-parameter. k=1 makes force ~1000x normal
    # in one step -> cart reaches max speed instantly, no momentum needed.
    if name == 'CartPole':
        tunable = {
            'masspole': IncrementUpdate(scheduler_1, k=5e-4),
            'gravity':  RandomWalk(scheduler_2, sigma=0.1),
        }
    elif name == 'MountainCar':
        tunable = {
            'gravity': IncrementUpdate(scheduler_1, k=1e-5),
            'force':   IncrementUpdate(scheduler_1, k=5e-6),
        }
    else:
        raise ValueError(f'unknown domain {domain}')

    ns_env = NSClassicControlWrapper(env, tunable, change_notification=True)
    return env, ns_env


def load_guidance_agent(domain, base_env):
    params = dict(cfg.agent)
    params['state_size']  = base_env.observation_space.shape[0]
    params['action_size'] = base_env.action_space.n
    params['model_path']  = DQN_MODEL_PATHS[domain]
    return DQNAgent(**params)


def make_agent(algorithm, planning_env, guidance_agent=None, noise_level=0.0):
    if algorithm == 'mcts':
        return MCTSAgent(planning_env)
    elif algorithm == 'inf_mcts':
        return InformedMCTSAgent(planning_env, guidance_agent, noise_level=noise_level)
    raise ValueError(f'unknown algorithm {algorithm}')


def run_config(domain, algorithm, noise_level):
    '''Run NUM_EPISODES episodes; return list of cumulative episodic rewards.'''
    base_env, ns_env = make_ns_env(domain)
    guidance_agent = load_guidance_agent(domain, base_env) if algorithm == 'inf_mcts' else None

    rewards = []
    for _ in tqdm(range(NUM_EPISODES),
                  desc=f'{domain} | {algorithm} | noise={noise_level}', leave=False):
        obs, _ = ns_env.reset()
        planning_env = ns_env.get_planning_env()
        agent = make_agent(algorithm, planning_env, guidance_agent, noise_level)

        episode_reward = 0.0
        done = truncated = False
        while not (done or truncated):
            action = agent.act(obs, planning_env)
            obs, reward, done, truncated, _ = ns_env.step(action)
            planning_env = ns_env.get_planning_env()
            episode_reward += reward
        rewards.append(episode_reward)

    base_env.close()
    return rewards


def experiment_labels(algorithm_filter='all'):
    '''Return list of (label, algorithm, noise_level) for selected filter.'''
    labels = []
    if algorithm_filter in ('all', 'mcts'):
        labels.append(('MCTS+UCT', 'mcts', None))
    if algorithm_filter in ('all', 'inf_mcts'):
        for nl in NOISE_LEVELS:
            labels.append((f'MCTS+informed UCT (noise={nl})', 'inf_mcts', nl))
    return labels


def run_all(algorithm_filter='all', domains=None):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    domains = domains or DOMAINS
    labels  = experiment_labels(algorithm_filter)
    results = {d: {} for d in domains}

    for domain in domains:
        for label, algorithm, noise_level in labels:
            repeats = []
            for r in range(NUM_REPEATS):
                print(f'[{domain}] {label}  repeat {r + 1}/{NUM_REPEATS}')
                repeats.append(run_config(domain, algorithm, noise_level or 0.0))
            arr = np.asarray(repeats, dtype=np.float32)   # (NUM_REPEATS, NUM_EPISODES)
            results[domain][label] = arr
            np.save(RESULTS_DIR / f'{domain}__{label}.npy', arr)
            plot_config(domain, label, noise_level, arr)   # plot after all repeats

    return results, labels, domains


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _line_plot(ax, arr, label, color=None):
    '''Plot mean reward on ax.'''
    mean = arr.mean(axis=0)
    x    = np.arange(1, mean.shape[0] + 1)
    ax.plot(x, mean, label=label, color=color)


def plot_config(domain, label, noise_level, arr):
    '''Save plot for one finished config (after all repeats).'''
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))
    _line_plot(ax, arr, label)
    ax.set_title(f'{label} — {domain}')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Cumulative reward')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    out = RESULTS_DIR / f'{domain}__{label}.png'
    fig.savefig(out, dpi=150, bbox_inches='tight')
    print(f'saved: {out}')
    plt.close(fig)


def plot_results(results, labels, domains=None):
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm

    domains = domains or DOMAINS
    colors = cm.tab10(np.linspace(0, 1, len(labels)))

    for domain in domains:
        # All algorithm labels on the same plot.
        fig, ax = plt.subplots(figsize=(10, 6))
        for (label, _, _), color in zip(labels, colors):
            _line_plot(ax, results[domain][label], label, color)
        ax.set_title(f'Cumulative episodic reward — {domain}')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Cumulative reward')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        out = RESULTS_DIR / f'{domain}_combined.png'
        fig.savefig(out, dpi=150, bbox_inches='tight')
        print(f'saved: {out}')
        plt.close(fig)


# ---------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MCTS experiments — full run.')
    parser.add_argument(
        '--algorithm', '-a',
        choices=['all', 'mcts', 'inf_mcts'],
        default='all',
        help='Algorithm to run (default: all)',
    )
    parser.add_argument(
        '--domain', '-d',
        choices=['all', 'cartpole', 'mountaincar'],
        default='all',
        help='Domain to run (default: all)',
    )
    parser.add_argument(
        '--render', '-r',
        action='store_true',
        help='Render environment visually during episodes (slow)',
    )
    args = parser.parse_args()

    if args.render:
        RENDER_MODE = 'human'

    domains = DOMAINS if args.domain == 'all' else [DOMAIN_MAP[args.domain]]

    results, labels, domains = run_all(args.algorithm, domains)
    plot_results(results, labels, domains)
