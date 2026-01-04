import gymnasium as gym
import numpy as np
import random
from gymnasium import spaces
from sim_class import Simulation 

class OT2Env(gym.Env):
    def __init__(self, render=False):
        super().__init__()
        # Initializing simulation with 1 agent
        self.sim = Simulation(num_agents=1, render=render)
        
        # Action space: 3 coordinates (x, y, z)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
        # Observation space: 3 coordinates (x, y, z) relative distance
        self.observation_space = spaces.Box(low=-10.0, high=10.0, shape=(3,), dtype=np.float32)
        
        # Default goal
        self.goal_pos = np.array([0.18, 0.18, 0.10], dtype=np.float32)
        
        # Addiction Maximus Rewards
        self.tiers = [0.100, 0.050, 0.025, 0.015, 0.010, 0.005, 0.002, 0.001]
        self.tier_rewards = [5, 10, 25, 50, 100, 250, 500, 1000]
        self.unlocked_tiers = [False] * len(self.tiers)
        
        self.stagnation_timer = 0 
        self.prev_dist = 0
        self.steps = 0

    def _get_pipette_pos(self, sim_output):
        """Standardizes dictionary output from Simulation into a numpy array."""
        if isinstance(sim_output, dict):
            return np.array(sim_output.get('pipette_position', [0,0,0]), dtype=np.float32)
        return np.array(sim_output, dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.steps = 0
        self.stagnation_timer = 0 
        self.unlocked_tiers = [False] * len(self.tiers)
        
        # Reset returns a dict
        raw_obs = self.sim.reset()
        pipette_pos = self._get_pipette_pos(raw_obs)

        # Success Priming: 10% chance to spawn near goal to jumpstart learning
        if random.random() < 0.10:
            self.goal_pos = pipette_pos + np.random.uniform(-0.005, 0.005, size=3)
        else:
            self.goal_pos = np.array([0.18, 0.18, 0.10], dtype=np.float32)

        delta = self.goal_pos - pipette_pos
        self.prev_dist = np.linalg.norm(delta)
        
        # Return observation and empty info dict
        return (delta * 10.0).astype(np.float32), {}

    def step(self, action):
        self.steps += 1
        
        # IMPORTANT: Method name is run_agent
        raw_obs = self.sim.run_agent(action)
        pipette_pos = self._get_pipette_pos(raw_obs)
        
        delta = self.goal_pos - pipette_pos
        curr_dist = np.linalg.norm(delta)
        diff = self.prev_dist - curr_dist
        
        # --- REWARD LOGIC: ADDICTION MAXIMUS ---
        
        # 1. Amygdala Withdrawal (Penalty for standing still)
        if diff > 0.0002:
            reward = diff * 1500 
            self.stagnation_timer = 0 
        else:
            self.stagnation_timer += 1
            withdrawal_penalty = 0.3 * (1 + (self.stagnation_timer / 10.0))
            reward = -withdrawal_penalty

        # 2. Dopamine Staircase (Tiered bonuses)
        for i, threshold in enumerate(self.tiers):
            if curr_dist < threshold and not self.unlocked_tiers[i]:
                bonus = random.uniform(0, self.tier_rewards[i] * 0.2)
                reward += self.tier_rewards[i] + bonus
                self.unlocked_tiers[i] = True

        # 3. Magnetism (Exponential pull when close)
        reward += 0.05 / (curr_dist + 0.0005)

        # Termination Check (0.8mm)
        terminated = bool(curr_dist < 0.0008)
        if terminated: 
            reward += 1000.0
            
        self.prev_dist = curr_dist
        truncated = self.steps >= 1000
        
        return (delta * 10.0).astype(np.float32), float(reward), terminated, truncated, {}

    def render(self):
        pass

    def close(self):
        self.sim.close()