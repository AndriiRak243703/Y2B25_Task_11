import numpy as np
import gymnasium as gym
from gymnasium import spaces
from sim_class import Simulation  # Your provided sim_class.py

class OT2Env(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, render=False):
        super().__init__()
        self.render_mode = "human" if render else None
        # ✅ Pass num_agents=1 explicitly
        self.sim = Simulation(num_agents=1, render=render, rgb_array=False)

        # Action space: [vx, vy, vz, drop] — but we'll ignore 'drop' for now
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)

        # Observation: pipette tip position [x, y, z]
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32)

        self.max_vel = 0.05  # m/s — fine control
        self.distance_threshold = 0.001  # 1 mm success
        self._step_count = 0
        self.max_episode_steps = 1000

    def _get_obs(self):
        states = self.sim.get_states()
        # Get pipette position of the first (and only) robot
        pipette_pos = list(states.values())[0]["pipette_position"]
        return np.array(pipette_pos, dtype=np.float32)

    def _calculate_reward(self, current_pos, distance_to_goal):
        """
        IMPROVED REWARD FOR 1MM PRECISION
        - Dense negative distance scaled to ~[-10, 0]
        - +10 bonus on success (<1mm)
        - Small step penalty
        """
        if distance_to_goal < self.distance_threshold:
            reward = 10.0
        else:
            reward = -distance_to_goal * 1000  # e.g., 5mm → -5.0
            reward -= 0.01  # discourage long episodes
        return float(reward)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            np.random.seed(seed)

        self.sim.reset(num_agents=1)
        obs = self._get_obs()

        # Generate goal within 1–5 cm
        rand_dist = self.np_random.uniform(0.01, 0.05)
        direction = self.np_random.uniform(-1, 1, size=3)
        direction[2] = abs(direction[2])  # keep z ≥ 0
        direction /= np.linalg.norm(direction) + 1e-8

        self.goal = obs + direction * rand_dist
        self.goal[0] = np.clip(self.goal[0], -0.25, 0.25)
        self.goal[1] = np.clip(self.goal[1], -0.25, 0.25)
        self.goal[2] = np.clip(self.goal[2], 0.01, 0.30)

        self._step_count = 0
        return obs, {}

    def step(self, action):
        self._step_count += 1

        # Convert action to format expected by sim: [[vx, vy, vz, drop]]
        # We fix drop=0 (no dispensing)
        full_action = [[float(action[0]), float(action[1]), float(action[2]), 0.0]]
        scaled_action = [
            [a * self.max_vel for a in agent_action]
            for agent_action in full_action
        ]

        self.sim.run(scaled_action, num_steps=1)
        pos = self._get_obs()
        distance = float(np.linalg.norm(pos - self.goal))

        # Boundary check
        out_of_bounds = (
            (pos[2] < 0.0) or
            not (-0.3 < pos[0] < 0.3) or
            not (-0.3 < pos[1] < 0.3) or
            (pos[2] > 0.4)
        )

        if out_of_bounds:
            reward = -20.0
            terminated = True
            is_success = 0.0
        elif distance < self.distance_threshold:
            reward = 10.0
            terminated = True
            is_success = 1.0
        else:
            reward = self._calculate_reward(pos, distance)
            terminated = False
            is_success = 0.0

        truncated = self._step_count >= self.max_episode_steps
        info = {
            "is_success": is_success,
            "dist_mm": distance * 1000.0
        }

        return pos, reward, terminated, truncated, info

    def compute_reward(self, achieved_goal, desired_goal, info):
        distance = np.linalg.norm(achieved_goal - desired_goal, axis=-1)
        reward = np.where(
            distance < self.distance_threshold,
            10.0,
            -distance * 1000 - 0.01
        )
        return reward.astype(np.float32)

    def close(self):
        self.sim.close()