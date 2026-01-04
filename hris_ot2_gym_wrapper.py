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

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.steps = 0
        self.stagnation_timer = 0 
        self.unlocked_tiers = [False] * len(self.tiers)
        
        # 1. Reset the simulation (returns a dictionary)
        reset_info = self.sim.reset() 
        
        # 2. Extract the position array from the dict 
        # In sim_class.py, this is usually under the 'pipette_position' key
        if isinstance(reset_info, dict) and 'pipette_position' in reset_info:
            pipette_pos = np.array(reset_info['pipette_position'], dtype=np.float32)
        else:
            # Fallback in case your sim_class returns the array directly
            pipette_pos = np.array(reset_info, dtype=np.float32)

        # 3. SUCCESS PRIMING Logic
        if random.random() < 0.10:
            self.goal_pos = pipette_pos + np.random.uniform(-0.005, 0.005, size=3)
        else:
            self.goal_pos = np.array([0.18, 0.18, 0.10], dtype=np.float32)

        # 4. Calculate initial distance
        delta = self.goal_pos - pipette_pos
        self.prev_dist = np.linalg.norm(delta)
        
        # Standard Gymnasium return: (observation, info_dict)
        return (delta * 10.0).astype(np.float32), {}