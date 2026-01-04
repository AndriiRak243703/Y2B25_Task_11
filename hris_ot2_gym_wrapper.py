import gymnasium as gym
from gymnasium import spaces
import numpy as np
from sim_class import Simulation
import random

class OT2Env(gym.Env):
    def __init__(self, render=False):
        super(OT2Env, self).__init__()
        # Initializing simulation with 1 agent
        self.sim = Simulation(num_agents=1, render=render)
        
        # Action space: 3 velocities [x, y, z] normalized to [-1, 1]
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
        
        # Observation space: 6D [current_x, current_y, current_z, goal_x, goal_y, goal_z]
        # Using 6D (friend's logic) is more stable for remote training
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(6,), dtype=np.float32)
        
        # OT-2 Workspace Bounds for normalization
        self.workspace_low = np.array([-0.1871, -0.1706, 0.1195], dtype=np.float32)
        self.workspace_high = np.array([0.2532, 0.2197, 0.2897], dtype=np.float32)
        
        # Addiction Maximus Reward Configuration
        self.tiers = [0.100, 0.050, 0.025, 0.015, 0.010, 0.005, 0.002, 0.001]
        self.tier_rewards = [5, 10, 25, 50, 100, 250, 500, 1000]
        self.unlocked_tiers = [False] * len(self.tiers)
        
        self.goal_pos = np.array([0.18, 0.18, 0.10], dtype=np.float32)
        self.stagnation_timer = 0 
        self.prev_dist = 0
        self.steps = 0

    def _normalize_pos(self, pos):
        """Standardizes raw coordinates to a [-1, 1] range."""
        return 2.0 * (pos - self.workspace_low) / (self.workspace_high - self.workspace_low) - 1.0

    def _extract_pos(self, state_dict):
        """Safely pulls pipette position from the simulation dictionary."""
        robot_id = list(sorted(state_dict.keys()))[0]
        return np.array(state_dict[robot_id].get('pipette_position', [0,0,0]), dtype=np.float32)

    def reset(self, seed=None):
        super().reset(seed=seed)
        self.steps = 0
        self.stagnation_timer = 0
        self.unlocked_tiers = [False] * len(self.tiers)
        
        # Reset the simulation
        state_dict = self.sim.reset(num_agents=1)
        current_pos = self._extract_pos(state_dict)
        
        # Randomize goal occasionally to prevent over-fitting
        if random.random() < 0.2:
            self.goal_pos = np.random.uniform(self.workspace_low, self.workspace_high).astype(np.float32)
        else:
            self.goal_pos = np.array([0.18, 0.18, 0.10], dtype=np.float32)

        self.prev_dist = np.linalg.norm(current_pos - self.goal_pos)
        
        obs = np.concatenate([self._normalize_pos(current_pos), self._normalize_pos(self.goal_pos)])
        return obs.astype(np.float32), {}

    def step(self, action):
        self.steps += 1
        
        # Velocity scaling (Friend's logic)
        velocity = action * 2.0
        full_action = [[float(velocity[0]), float(velocity[1]), float(velocity[2]), 0.0]]
        
        # Execute in simulation
        state_dict = self.sim.run(full_action)
        current_pos = self._extract_pos(state_dict)
        
        curr_dist = np.linalg.norm(current_pos - self.goal_pos)
        diff = self.prev_dist - curr_dist
        
        # --- ADDICTION MAXIMUS REWARD LOGIC ---
        if diff > 0.0002:
            reward = diff * 1500 
            self.stagnation_timer = 0 
        else:
            self.stagnation_timer += 1
            reward = -(0.3 * (1 + (self.stagnation_timer / 10.0)))

        # Tiered Bonuses
        for i, threshold in enumerate(self.tiers):
            if curr_dist < threshold and not self.unlocked_tiers[i]:
                reward += self.tier_rewards[i]
                self.unlocked_tiers[i] = True

        terminated = bool(curr_dist < 0.001) # Success at 1mm
        if terminated: reward += 1000.0
        
        truncated = self.steps >= 1000
        self.prev_dist = curr_dist
        
        obs = np.concatenate([self._normalize_pos(current_pos), self._normalize_pos(self.goal_pos)])
        return obs.astype(np.float32), float(reward), terminated, truncated, {'distance': curr_dist}

    def close(self):
        self.sim.close()