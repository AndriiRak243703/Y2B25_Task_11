import numpy as np
import gymnasium as gym
from gymnasium import spaces
from sim_class import Simulation 

class OT2Env(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, render=False):
        super().__init__()
        self.render_mode = "human" if render else None
        
        # Initialize simulation
        self.sim = Simulation(num_agents=1, render=render, rgb_array=False)

        # Action space: [vx, vy, vz]
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)

        # Observation space: [pipette_x, pipette_y, pipette_z, goal_x, goal_y, goal_z]
        # Shape is 6 because we MUST tell the agent where the goal is.
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32)

        self.max_vel = 0.05 
        self.distance_threshold = 0.001 # 1mm success threshold
        self._step_count = 0
        self.max_episode_steps = 1000

    def _get_obs(self):
        states = self.sim.get_states()
        pipette_pos = np.array(list(states.values())[0]["pipette_position"], dtype=np.float32)
        
        # Concatenate current position (3) + goal position (3)
        return np.concatenate([pipette_pos, self.goal.astype(np.float32)])

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            np.random.seed(seed)

        self.sim.reset(num_agents=1)
        
        # 1. Initialize a dummy goal temporarily so we can get the first observation
        self.goal = np.zeros(3, dtype=np.float32)
        
        # 2. Get current pipette position
        states = self.sim.get_states()
        current_pos = np.array(list(states.values())[0]["pipette_position"])

        # 3. Generate a random goal 2cm - 10cm away from the current position
        rand_dist = self.np_random.uniform(0.02, 0.10)
        direction = self.np_random.uniform(-1, 1, size=3)
        direction /= (np.linalg.norm(direction) + 1e-8)
        
        self.goal = current_pos + direction * rand_dist
        
        # Clip goal to ensure it stays within the workspace
        self.goal[0] = np.clip(self.goal[0], -0.25, 0.25)
        self.goal[1] = np.clip(self.goal[1], -0.25, 0.25)
        self.goal[2] = np.clip(self.goal[2], 0.05, 0.25) # Keep above the floor

        self._step_count = 0
        return self._get_obs(), {}

    def step(self, action):
        self._step_count += 1

        # Action format for sim: [[vx, vy, vz, drop]]
        # We append 0.0 for the drop action
        full_action = [[float(action[0]), float(action[1]), float(action[2]), 0.0]]
        
        # Scale action by max velocity
        scaled_action = [
            [a * self.max_vel for a in agent_action]
            for agent_action in full_action
        ]

        self.sim.run(scaled_action, num_steps=1)
        
        # Get new state
        obs = self._get_obs()
        pipette_pos = obs[:3] # First 3 values are position
        distance = float(np.linalg.norm(pipette_pos - self.goal))

        # Check for success
        if distance < self.distance_threshold:
            reward = 10.0
            terminated = True
            is_success = 1.0
        else:
            # Dense reward: negative distance
            reward = -distance * 10.0 
            reward -= 0.01 # Small time penalty
            terminated = False
            is_success = 0.0

        truncated = self._step_count >= self.max_episode_steps
        
        info = {
            "is_success": is_success,
            "dist_mm": distance * 1000.0
        }

        return obs, float(reward), terminated, truncated, info

    def close(self):
        self.sim.close()