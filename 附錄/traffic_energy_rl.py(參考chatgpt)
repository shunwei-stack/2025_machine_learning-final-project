"""
Simplified City Traffic-Energy Simulation + RL Training Example
File: traffic_energy_rl.py
Dependencies:
  - numpy
  - gym (or gymnasium)
  - matplotlib
Optional:
  - stable_baselines3 (if available, PPO will be used; otherwise a fallback policy is used)

This script defines a simple Gym environment and provides two execution paths:
  - when stable_baselines3 is installed: train a PPO agent
  - when it is not available: use a deterministic heuristic fallback model

The file also includes lightweight tests that validate core behaviors.
"""

import math
import sys
import numpy as np
try:
    import gymnasium as gym
    from gym import spaces
except Exception:
    import gymnasium as gym
    from gymnasium import spaces
import matplotlib.pyplot as plt

HAS_SB3 = True
try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv
except Exception:
    HAS_SB3 = False


class CityTrafficEnergyEnv(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(self, seed=None):
        super().__init__()
        self.rng = np.random.RandomState(seed)
        high = np.array([1.0, 1.0, 1.0], dtype=np.float32)
        low = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)
        self.action_space = spaces.Discrete(4)
        self.max_steps = 24
        self.step_count = 0
        self.traffic_inertia = 0.7
        self.energy_inertia = 0.8
        self.noise_scale = 0.02
        self.alpha_emission = 1.0
        self.beta_commute = 1.0
        self.reset()

    def reset(self, *, seed=None, return_info=False, options=None):
        if seed is not None:
            self.rng = np.random.RandomState(seed)
        self.traffic_load = float(self.rng.uniform(0.3, 0.6))
        self.energy_demand = float(self.rng.uniform(0.3, 0.6))
        self.time_of_day = float(self.rng.uniform(0.0, 1.0))
        self.step_count = 0
        obs = np.array([self.traffic_load, self.energy_demand, self.time_of_day], dtype=np.float32)
        if return_info:
            return obs, {}
        return obs

    def step(self, action):
        action = int(np.asarray(action).squeeze())
        hour = int(self.time_of_day * 24) % 24
        demand_diurnal = 0.3 + 0.7 * math.exp(-((hour - 18) ** 2) / (2 * 6 ** 2))
        traffic_diurnal = 0.2 + 0.8 * math.exp(-((hour - 8) ** 2) / (2 * 3 ** 2))
        delta_traffic = 0.0
        delta_energy = 0.0
        if action == 0:
            delta_traffic -= 0.15
            delta_energy += 0.01
        elif action == 1:
            delta_energy -= 0.18
        elif action == 2:
            delta_energy -= 0.08
            delta_traffic += 0.06
        next_traffic = (
            self.traffic_inertia * self.traffic_load
            + (1 - self.traffic_inertia) * traffic_diurnal
            + delta_traffic
            + float(self.rng.normal(scale=self.noise_scale))
        )
        next_energy = (
            self.energy_inertia * self.energy_demand
            + (1 - self.energy_inertia) * demand_diurnal
            + delta_energy
            + float(self.rng.normal(scale=self.noise_scale))
        )
        self.traffic_load = float(np.clip(next_traffic, 0.0, 1.0))
        self.energy_demand = float(np.clip(next_energy, 0.0, 1.0))
        self.step_count += 1
        self.time_of_day = (self.time_of_day + 1.0 / 24.0) % 1.0
        commute_time = 1.0 + 3.0 * (self.traffic_load ** 1.5)
        emissions = 0.6 * self.energy_demand + 0.4 * self.traffic_load
        reward = - (self.alpha_emission * emissions + self.beta_commute * commute_time)
        done = bool(self.step_count >= self.max_steps)
        info = {
            'commute_time': commute_time,
            'emissions': emissions,
            'traffic_load': self.traffic_load,
            'energy_demand': self.energy_demand,
            'time_of_day': self.time_of_day,
        }
        obs = np.array([self.traffic_load, self.energy_demand, self.time_of_day], dtype=np.float32)
        return obs, float(reward), done, info


class FallbackModel:
    def __init__(self):
        pass

    def predict(self, observation, deterministic=True):
        obs = np.asarray(observation, dtype=np.float32).squeeze()
        if obs.ndim != 1 or obs.shape[0] != 3:
            obs = obs.flatten()
        traffic, energy, _ = float(obs[0]), float(obs[1]), float(obs[2])
        if traffic > 0.6:
            return 0, None
        if energy > 0.6:
            return 1, None
        if energy > 0.4:
            return 2, None
        return 3, None

    def save(self, path):
        return None


def train_agent(total_timesteps=200_000, seed=0):
    if HAS_SB3:
        env = DummyVecEnv([lambda: CityTrafficEnergyEnv(seed=seed)])
        model = PPO('MlpPolicy', env, verbose=1, seed=seed)
        model.learn(total_timesteps=total_timesteps)
        model.save('ppo_city_traffic_energy')
        return model
    else:
        return FallbackModel()


def _predict_action_from_model(model, observation):
    action, _ = model.predict(observation, deterministic=True)
    if isinstance(action, np.ndarray):
        action = int(action.squeeze())
    else:
        action = int(action)
    return action


def evaluate(model, episodes=20, seed=1):
    env = CityTrafficEnergyEnv(seed=seed)
    metrics = {'reward': [], 'commute_time': [], 'emissions': []}
    for _ in range(episodes):
        obs = env.reset()
        done = False
        ep_reward = 0.0
        last_info = None
        while not done:
            action = _predict_action_from_model(model, obs)
            obs, r, done, info = env.step(action)
            last_info = info
            ep_reward += r
        metrics['reward'].append(ep_reward)
        metrics['commute_time'].append(last_info['commute_time'])
        metrics['emissions'].append(last_info['emissions'])
    return metrics


def plot_metrics(metrics, title_prefix='Result'):
    plt.figure()
    plt.plot(metrics['reward'], marker='o')
    plt.title(f'{title_prefix}: Episode cumulative reward')
    plt.xlabel('episode')
    plt.ylabel('cumulative reward')
    plt.grid(True)
    plt.figure()
    plt.plot(metrics['commute_time'], marker='o')
    plt.title(f'{title_prefix}: Final commute time (per episode)')
    plt.xlabel('episode')
    plt.ylabel('commute_time')
    plt.grid(True)
    plt.figure()
    plt.plot(metrics['emissions'], marker='o')
    plt.title(f'{title_prefix}: Final emissions (per episode)')
    plt.xlabel('episode')
    plt.ylabel('emissions')
    plt.grid(True)
    plt.show()


def baseline_evaluation(episodes=20, seed=2):
    env = CityTrafficEnergyEnv(seed=seed)
    metrics = {'reward': [], 'commute_time': [], 'emissions': []}
    for _ in range(episodes):
        obs = env.reset()
        done = False
        ep_reward = 0.0
        last_info = None
        while not done:
            obs, r, done, info = env.step(3)
            last_info = info
            ep_reward += r
        metrics['reward'].append(ep_reward)
        metrics['commute_time'].append(last_info['commute_time'])
        metrics['emissions'].append(last_info['emissions'])
    return metrics


def _test_env_step():
    env = CityTrafficEnergyEnv(seed=0)
    obs = env.reset()
    assert obs.shape == (3,)
    for a in range(4):
        new_obs, r, done, info = env.step(a)
        assert isinstance(r, float)
        assert 'emissions' in info
    print('test_env_step passed')


def _test_baseline_and_evaluate():
    model = FallbackModel()
    base = baseline_evaluation(episodes=5, seed=0)
    assert len(base['reward']) == 5
    metrics = evaluate(model, episodes=5, seed=0)
    assert len(metrics['reward']) == 5
    print('test_baseline_and_evaluate passed')


def run_tests():
    _test_env_step()
    _test_baseline_and_evaluate()


if __name__ == '__main__':
    run_tests()
    model = train_agent(total_timesteps=80_000, seed=42)
    rl_metrics = evaluate(model, episodes=20, seed=100)
    base_metrics = baseline_evaluation(episodes=20, seed=100)
    def summarize(name, m):
        print(f"--- {name} ---")
        print('reward mean/std:', np.mean(m['reward']), np.std(m['reward']))
        print('commute_time mean/std:', np.mean(m['commute_time']), np.std(m['commute_time']))
        print('emissions mean/std:', np.mean(m['emissions']), np.std(m['emissions']))
    summarize('RL agent', rl_metrics)
    summarize('Baseline (do nothing)', base_metrics)
    plot_metrics(rl_metrics, title_prefix='RL agent')
    plot_metrics(base_metrics, title_prefix='Baseline')
    if HAS_SB3:
        print('\nSaved model: ppo_city_traffic_energy.zip')
    else:
        print('\nstable_baselines3 not available; used fallback policy.')
