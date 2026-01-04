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
        
        # Success Parameters
        self.goal_pos = np.array([0.18, 0.18, 0.10], dtype=np.float32)
        self.settle_steps = 0
        self.required_settle = 15 # Harder to "win"
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
        
        state_dict = self.sim.reset(num_agents=1)
        current_pos = self._extract_pos(state_dict)
        
        # --- THE "HOOK": SUCCESS PRIMING ---
        # 5% of the time, spawn the goal RIGHT NEXT to the robot so it gets a "Free Win"
        if random.random() < 0.05:
            self.goal_pos = current_pos + np.random.uniform(-0.01, 0.01, size=3)
        else:
            self.goal_pos = np.array([0.18, 0.18, 0.10], dtype=np.float32)
            
        obs = np.concatenate([self._normalize_pos(current_pos), self._normalize_pos(self.goal_pos)])
        return obs.astype(np.float32), {}

    def step(self, action):
        self.steps += 1
        velocity = action * 2.0
        speed = np.linalg.norm(velocity)
        
        state_dict = self.sim.run([[float(velocity[0]), float(velocity[1]), float(velocity[2]), 0.0]])
        current_pos = self._extract_pos(state_dict)
        curr_dist = np.linalg.norm(current_pos - self.goal_pos)
        
        # ========================================================================
        # THE GACHA REWARD ENGINE (ADDICTION MAXIMUS)
        # ========================================================================
        reward = 0
        
        # 1. THE "SLOT MACHINE" NOISE (Fake Dopamine)
        # Gives a tiny random reward just for moving, keeps the agent "pulling the lever"
        if speed > 0.01:
            reward += random.uniform(0, 0.05) 

        # 2. THE GRAVITY WELL (The Hunger)
        # Harsh penalty for being far away. 100mm = -3.1pts, 1mm = -0.3pts
        reward -= 10.0 * (curr_dist ** 0.5)

        # 3. NEAR-MISS BONUSES (The Tease)
        # Rewards "almost winning" to keep it trying
        if 0.005 < curr_dist < 0.015:
            reward += 0.5 # "So close!"

        # 4. THE 1mm JACKPOT (The Real Win)
        if curr_dist < 0.001:
            self.settle_steps += 1
            # Exponentially increasing dopamine during settling
            reward += (50.0 * self.settle_steps)
            reward -= (speed * 20.0) # HEAVY penalty for moving while winning
        else:
            self.settle_steps = 0

        # 5. THE ULTIMATE PAYOUT
        terminated = bool(self.settle_steps >= self.required_settle)
        if terminated:
            reward += 10000.0 # THE GRAND PRIZE
        
        truncated = self.steps >= 600 # Faster episodes for more "rounds" of gambling
        
        obs = np.concatenate([self._normalize_pos(current_pos), self._normalize_pos(self.goal_pos)])
        return obs.astype(np.float32), float(reward), terminated, truncated, {'dist_mm': curr_dist * 1000}