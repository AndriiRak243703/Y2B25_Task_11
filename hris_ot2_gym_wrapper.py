import gymnasium as gym
import numpy as np
import random
from gymnasium import spaces
from sim_class import Simulation # CITED: - Changed from OT2Sim to match project sim_class.py

class OT2Env(gym.Env):
    def __init__(self, render=False):
        super().__init__()
        # Ensure we use the provided sim_class.py as per task requirements
        self.sim = Simulation(num_agents=1, render=render) # CITED:
        
        # Action space: x, y, z movements
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
        
        # Observation space: 3D distance to goal
        self.observation_space = spaces.Box(low=-10.0, high=10.0, shape=(3,), dtype=np.float32)
        
        # Initial Goal
        self.goal_pos = np.array([0.18, 0.18, 0.10], dtype=np.float32)
        
        # Reward Tiers (The "Dopamine Staircase")
        self.tiers = [0.100, 0.050, 0.025, 0.015, 0.010, 0.005, 0.002, 0.001]
        self.tier_rewards = [5, 10, 25, 50, 100, 250, 500, 1000]
        self.unlocked_tiers = [False] * len(self.tiers)
        
        # BIOLOGICAL ADDICTION FEATURES
        self.stagnation_timer = 0 
        self.prev_dist = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.steps = 0
        self.stagnation_timer = 0 
        self.unlocked_tiers = [False] * len(self.tiers)
        
        # Reset the simulation and get initial pipette position
        # sim_class.py reset() usually returns coordinates
        pipette_pos = self.sim.reset() 

        # SUCCESS PRIMING (The "Shortcut" Effect)
        # Occasionally start very close to the goal to "teach" the agent what success feels like
        if random.random() < 0.10:
            self.goal_pos = pipette_pos + np.random.uniform(-0.005, 0.005, size=3)
        else:
            # Standard goal position within the robot's workspace
            self.goal_pos = np.array([0.18, 0.18, 0.10], dtype=np.float32)

        delta = self.goal_pos - pipette_pos
        self.prev_dist = np.linalg.norm(delta)
        
        # Standardize return observation
        return (delta * 10.0).astype(np.float32), {}

    def step(self, action):
        self.steps += 1
        # Execute action in the simulation
        pipette_pos = self.sim.run_step(action) 
        
        delta = self.goal_pos - pipette_pos
        curr_dist = np.linalg.norm(delta)
        diff = self.prev_dist - curr_dist
        
        # --- REWARD LOGIC ---
        
        # 1. THE AMYGDALA WITHDRAWAL (Anti-Stagnation)
        if diff > 0.0002: # Significant progress toward goal
            reward = diff * 1500 
            self.stagnation_timer = 0 
        else:
            self.stagnation_timer += 1
            # Penalty increases the longer we stay still
            withdrawal_penalty = 0.3 * (1 + (self.stagnation_timer / 10.0))
            reward = -withdrawal_penalty

        # 2. THE DOPAMINE STAIRCASE (Tiered Rewards)
        for i, threshold in enumerate(self.tiers):
            if curr_dist < threshold and not self.unlocked_tiers[i]:
                # Variable Reward Bonus to prevent the agent from getting "stuck" on specific thresholds
                bonus = random.uniform(0, self.tier_rewards[i] * 0.2)
                reward += self.tier_rewards[i] + bonus
                self.unlocked_tiers[i] = True

        # 3. MAGNETISM (Exponential Reward for proximity)
        reward += 0.05 / (curr_dist + 0.0005)

        # Termination condition (Within 0.8mm)
        terminated = bool(curr_dist < 0.0008)
        if terminated: 
            reward += 1000.0
            
        self.prev_dist = curr_dist
        
        # Truncation condition (Max steps per episode)
        truncated = self.steps >= 1000
        
        return (delta * 10.0).astype(np.float32), float(reward), terminated, truncated, {}