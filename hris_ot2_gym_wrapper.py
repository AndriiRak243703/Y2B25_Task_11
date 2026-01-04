import gymnasium as gym
import numpy as np
import random
from gymnasium import spaces
from sim_class import Simulation 

class OT2Env(gym.Env):
    def __init__(self, render=False):
        super().__init__()
        self.sim = Simulation(num_agents=1, render=render)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-10.0, high=10.0, shape=(3,), dtype=np.float32)
        self.goal_pos = np.array([0.18, 0.18, 0.10], dtype=np.float32)
        
        self.tiers = [0.100, 0.050, 0.025, 0.015, 0.010, 0.005, 0.002, 0.001]
        self.tier_rewards = [5, 10, 25, 50, 100, 250, 500, 1000]
        self.unlocked_tiers = [False] * len(self.tiers)
        
        self.stagnation_timer = 0 
        self.prev_dist = 0

    def _get_pipette_pos(self, sim_output):
        """Helper to extract position from simulation dictionary safely."""
        if isinstance(sim_output, dict):
            # The key 'pipette_position' is standard for the group sim_class
            return np.array(sim_output.get('pipette_position', [0,0,0]), dtype=np.float32)
        return np.array(sim_output, dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.steps = 0
        self.stagnation_timer = 0 
        self.unlocked_tiers = [False] * len(self.tiers)
        
        # Reset returns a dictionary
        raw_obs = self.sim.reset()
        pipette_pos = self._get_pipette_pos(raw_obs)

        if random.random() < 0.10:
            self.goal_pos = pipette_pos + np.random.uniform(-0.005, 0.005, size=3)
        else:
            self.goal_pos = np.array([0.18, 0.18, 0.10], dtype=np.float32)

        delta = self.goal_pos - pipette_pos
        self.prev_dist = np.linalg.norm(delta)
        return (delta * 10.0).astype(np.float32), {}

    def step(self, action):
        self.steps += 1
        # sim.run_step also returns a dictionary
        raw_obs = self.sim.run_step(action)
        pipette_pos = self._get_pipette_pos(raw_obs)
        
        delta = self.goal_pos - pipette_pos
        curr_dist = np.linalg.norm(delta)
        diff = self.prev_dist - curr_dist
        
        # 1. AMYGDALA WITHDRAWAL
        if diff > 0.0002:
            reward = diff * 1500 
            self.stagnation_timer = 0 
        else:
            self.stagnation_timer += 1
            withdrawal_penalty = 0.3 * (1 + (self.stagnation_timer / 10.0))
            reward = -withdrawal_penalty

        # 2. DOPAMINE STAIRCASE
        for i, threshold in enumerate(self.tiers):
            if curr_dist < threshold and not self.unlocked_tiers[i]:
                bonus = random.uniform(0, self.tier_rewards[i] * 0.2)
                reward += self.tier_rewards[i] + bonus
                self.unlocked_tiers[i] = True

        # 3. MAGNETISM
        reward += 0.05 / (curr_dist + 0.0005)

        terminated = bool(curr_dist < 0.0008)
        if terminated: reward += 1000.0
            
        self.prev_dist = curr_dist
        truncated = self.steps >= 1000
        
        # Explicitly cast reward to float for SB3 compatibility
        return (delta * 10.0).astype(np.float32), float(reward), terminated, truncated, {}