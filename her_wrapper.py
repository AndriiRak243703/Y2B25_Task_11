import numpy as np
import gymnasium as gym
from gymnasium import spaces
from sim_class import Simulation 

class OT2Env(gym.Env):
    def __init__(self, render=False):
        super().__init__()
        self.sim = Simulation(num_agents=1, render=render, rgb_array=False)

        # Action space: [vx, vy, vz]
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)

        # Observation: [pipette_x, y, z, goal_x, y, z] (Shape 6)
        # The agent NEEDS to see the goal to learn how to reach it.
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32)

        self.max_vel = 0.05 
        self.distance_threshold = 0.001 # 1mm
        self._step_count = 0

    def _get_obs(self):
        states = self.sim.get_states()
        pipette_pos = np.array(list(states.values())[0]["pipette_position"], dtype=np.float32)
        # Concatenate current position and goal position
        return np.concatenate([pipette_pos, self.goal.astype(np.float32)])

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.sim.reset(num_agents=1)
        
        # 1. Initialize a dummy goal first so _get_obs doesn't crash
        self.goal = np.zeros(3, dtype=np.float32)
        
        # 2. Get current position to calculate a relative goal
        states = self.sim.get_states()
        current_pos = np.array(list(states.values())[0]["pipette_position"])

        # 3. Generate random goal
        rand_dist = self.np_random.uniform(0.02, 0.10) # 2cm to 10cm away
        direction = self.np_random.uniform(-1, 1, size=3)
        direction /= (np.linalg.norm(direction) + 1e-8)
        
        self.goal = current_pos + direction * rand_dist
        self.goal[2] = np.clip(self.goal[2], 0.05, 0.25) # Keep height safe

        self._step_count = 0
        return self._get_obs(), {}

    def step(self, action):
        self._step_count += 1
        # Scale action to max velocity
        scaled_action = [[float(a * self.max_vel) for a in action] + [0.0]]
        self.sim.run(scaled_action, num_steps=1)
        
        obs = self._get_obs()
        pipette_pos = obs[:3]
        distance = float(np.linalg.norm(pipette_pos - self.goal))

        # Reward logic
        is_success = distance < self.distance_threshold
        reward = 10.0 if is_success else -distance
        
        terminated = is_success
        truncated = self._step_count >= 1000
        
        return obs, float(reward), terminated, truncated, {"is_success": float(is_success), "dist_mm": distance*1000}

    def close(self):
        self.sim.close()