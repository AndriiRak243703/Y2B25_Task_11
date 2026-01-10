import numpy as np
import gymnasium as gym
from gymnasium import spaces
from sim_class import Simulation  # Provided simulation class

class OT2Env(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, render=False):
        super().__init__()
        self.render_mode = "human" if render else None
        self.sim = Simulation(render=render)

        # Action: [vx, vy, vz] → normalized to [-1, 1], scaled internally
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)

        # Observation: [x, y, z] pipette tip position
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32)

        # Task parameters
        self.max_vel = 0.05  # m/s — reduced for fine control
        self.distance_threshold = 0.001  # 1 mm success threshold
        self.max_episode_steps = 1000
        self._step_count = 0

    def _get_obs(self):
        pos, _ = self.sim.get_state()
        return pos.astype(np.float32)

    def _calculate_reward(self, current_pos, distance_to_goal):
        """
        IMPROVED REWARD FOR 1MM PRECISION:
        - Dense negative distance scaled to ~[-10, 0]
        - +10.0 bonus on success (<1mm)
        - Small step penalty to discourage inefficiency
        """
        if distance_to_goal < self.distance_threshold:
            reward = 10.0
        else:
            reward = -distance_to_goal * 1000  # e.g., 5mm → -5.0
            reward -= 0.01  # slight penalty per step
        return float(reward)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            np.random.seed(seed)

        self.sim.reset()
        obs = self._get_obs()

        # Generate goal near robot workspace
        rand_dist = self.np_random.uniform(0.01, 0.05)  # 1–5 cm for easier learning
        direction = self.np_random.uniform(-1, 1, size=3)
        direction[2] = abs(direction[2])  # bias upward (z ≥ 0)
        direction /= np.linalg.norm(direction) + 1e-8

        self.goal = obs + direction * rand_dist

        # Clamp goal to safe workspace
        self.goal[0] = np.clip(self.goal[0], -0.25, 0.25)
        self.goal[1] = np.clip(self.goal[1], -0.25, 0.25)
        self.goal[2] = np.clip(self.goal[2], 0.01, 0.30)

        self._step_count = 0
        return obs, {}

    def step(self, action):
        self._step_count += 1

        # Scale action to physical velocity
        scaled_action = np.clip(action, -1.0, 1.0) * self.max_vel
        self.sim.run(scaled_action)

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
            "dist_mm": distance * 1000.0  # for logging in mm
        }

        return pos, reward, terminated, truncated, info

    def compute_reward(self, achieved_goal, desired_goal, info):
        """
        Required for compatibility (e.g., evaluation). Must match step() logic.
        Note: This env does NOT use goal-conditioned observations, so this is rarely called.
        But we include it for safety.
        """
        distance = np.linalg.norm(achieved_goal - desired_goal, axis=-1)
        reward = np.where(
            distance < self.distance_threshold,
            10.0,
            -distance * 1000 - 0.01
        )
        return reward.astype(np.float32)

    def render(self):
        pass  # Rendering handled by Simulation if GUI mode

    def close(self):
        self.sim.close() if hasattr(self.sim, 'close') else None