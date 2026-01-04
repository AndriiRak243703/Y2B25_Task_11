import gymnasium as gym
from gymnasium import spaces
import numpy as np
from sim_class import Simulation
import random

class OT2Env(gym.Env):
    def __init__(self, render=False):
        super(OT2Env, self).__init__()
        self.sim = Simulation(num_agents=1, render=render)
        
        # Action space: 3 velocities [x, y, z] normalized to [-1, 1]
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
        
        # Observation space: 6D [curr_x, curr_y, curr_z, goal_x, goal_y, goal_z]
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(6,), dtype=np.float32)
        
        # Workspace Bounds
        self.workspace_low = np.array([-0.1871, -0.1706, 0.1195], dtype=np.float32)
        self.workspace_high = np.array([0.2532, 0.2197, 0.2897], dtype=np.float32)
        
        # Success and Settling Configuration
        self.goal_pos = np.array([0.18, 0.18, 0.10], dtype=np.float32)
        self.target_threshold = 0.001 # 1mm
        self.settle_steps = 0
        self.required_settle = 10
        
        # Reward Tiers (Dopamine Staircase)
        self.tiers = [0.100, 0.050, 0.025, 0.015, 0.010, 0.005, 0.002, 0.001]
        self.tier_rewards = [10, 20, 50, 100, 200, 500, 1000, 2000]
        self.unlocked_tiers = [False] * len(self.tiers)
        
        self.steps = 0

    def _normalize_pos(self, pos):
        return 2.0 * (pos - self.workspace_low) / (self.workspace_high - self.workspace_low) - 1.0

    def _extract_pos(self, state_dict):
        robot_id = list(sorted(state_dict.keys()))[0]
        return np.array(state_dict[robot_id].get('pipette_position', [0,0,0]), dtype=np.float32)

    def reset(self, seed=None):
        super().reset(seed=seed)
        self.steps = 0
        self.settle_steps = 0
        self.unlocked_tiers = [False] * len(self.tiers)
        
        state_dict = self.sim.reset(num_agents=1)
        current_pos = self._extract_pos(state_dict)
        
        # Fixed goal for consistency during precision training
        self.goal_pos = np.array([0.18, 0.18, 0.10], dtype=np.float32)
        
        obs = np.concatenate([self._normalize_pos(current_pos), self._normalize_pos(self.goal_pos)])
        return obs.astype(np.float32), {}

    def step(self, action):
        self.steps += 1
        
        # Action Scaling (Speed control)
        velocity = action * 2.0
        speed = np.linalg.norm(velocity)
        full_action = [[float(velocity[0]), float(velocity[1]), float(velocity[2]), 0.0]]
        
        state_dict = self.sim.run(full_action)
        current_pos = self._extract_pos(state_dict)
        
        curr_dist = np.linalg.norm(current_pos - self.goal_pos)
        
        # --- PRECISION REWARD LOGIC ---
        
        # 1. Gravity Well: Punish distance exponentially (forces agent to move)
        reward = -15.0 * (curr_dist ** 0.4)
        
        # 2. Velocity Leash: Punish speed when very close (forces slowing down)
        if curr_dist < 0.015: # Within 1.5cm
            reward -= (speed * 8.0)
            
        # 3. Settling Mechanics
        if curr_dist < self.target_threshold:
            self.settle_steps += 1
            reward += 20.0 * self.settle_steps # Bonus for staying inside
        else:
            self.settle_steps = 0 # Reset if it flies out

        # 4. Tiered Bonuses (The "Jackpots")
        for i, threshold in enumerate(self.tiers):
            if curr_dist < threshold and not self.unlocked_tiers[i]:
                reward += self.tier_rewards[i]
                self.unlocked_tiers[i] = True

        # Termination: Must be settled to finish
        terminated = bool(self.settle_steps >= self.required_settle)
        if terminated:
            reward += 5000.0 # The ultimate jackpot
        
        truncated = self.steps >= 1000
        
        obs = np.concatenate([self._normalize_pos(current_pos), self._normalize_pos(self.goal_pos)])
        return obs.astype(np.float32), float(reward), terminated, truncated, {'dist_mm': curr_dist * 1000}

    def close(self):
        self.sim.close()