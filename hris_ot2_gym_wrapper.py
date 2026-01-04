import gymnasium as gym
from gymnasium import spaces
import numpy as np
from sim_class import Simulation
import random

class OT2Env(gym.Env):
    def __init__(self, render=False):
        super(OT2Env, self).__init__()
        self.sim = Simulation(num_agents=1, render=render)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(6,), dtype=np.float32)
        
        self.workspace_low = np.array([-0.1871, -0.1706, 0.1195], dtype=np.float32)
        self.workspace_high = np.array([0.2532, 0.2197, 0.2897], dtype=np.float32)
        
        self.goal_pos = np.array([0.18, 0.18, 0.10], dtype=np.float32)
        self.prev_dist = 0
        self.withdrawal_timer = 0
        self.steps = 0

    def _normalize_pos(self, pos):
        return 2.0 * (pos - self.workspace_low) / (self.workspace_high - self.workspace_low) - 1.0

    def _extract_pos(self, state_dict):
        robot_id = list(sorted(state_dict.keys()))[0]
        return np.array(state_dict[robot_id].get('pipette_position', [0,0,0]), dtype=np.float32)

    def reset(self, seed=None):
        super().reset(seed=seed)
        self.steps = 0
        self.withdrawal_timer = 0
        
        state_dict = self.sim.reset(num_agents=1)
        current_pos = self._extract_pos(state_dict)
        self.prev_dist = np.linalg.norm(current_pos - self.goal_pos)
        
        obs = np.concatenate([self._normalize_pos(current_pos), self._normalize_pos(self.goal_pos)])
        return obs.astype(np.float32), {}

    def step(self, action):
        self.steps += 1
        # Action is clamped to prevent the 3.6e+12 explosion
        velocity = np.clip(action, -1.0, 1.0) * 2.0
        
        state_dict = self.sim.run([[float(velocity[0]), float(velocity[1]), float(velocity[2]), 0.0]])
        current_pos = self._extract_pos(state_dict)
        curr_dist = np.linalg.norm(current_pos - self.goal_pos)
        
        # --- THE ADDICTION ENGINE ---
        reward = 0
        progress = self.prev_dist - curr_dist

        # 1. THE DOPAMINE HIT (Positive Reinforcement)
        # Give a massive hit for any progress toward the goal
        if progress > 0.0005:
            reward += (progress * 5000) 
            self.withdrawal_timer = 0 # Addiction satisfied
        else:
            # 2. WITHDRAWAL SYNDROME (The Penalty)
            # Penalize for stagnation or moving away. Penalty grows over time!
            self.withdrawal_timer += 1
            withdrawal_penalty = 0.5 * (1.1 ** self.withdrawal_timer) 
            reward -= min(withdrawal_penalty, 50) # Cap penalty to keep brain stable

        # 3. THE "WIN" (1mm target)
        if curr_dist < 0.001:
            reward += 100 # Sustained hit for being on target
            terminated = True
        else:
            terminated = False

        # 4. SAFETY CLAMP
        reward = np.clip(reward, -100, 100)
        
        self.prev_dist = curr_dist
        truncated = self.steps >= 1000
        
        obs = np.concatenate([self._normalize_pos(current_pos), self._normalize_pos(self.goal_pos)])
        return obs.astype(np.float32), float(reward), terminated, truncated, {'dist_mm': curr_dist * 1000}