import numpy as np
import gymnasium as gym
from gymnasium import spaces
from sim_class import Simulation 

class OT2Env(gym.Env):
    def __init__(self, render=False):
        super().__init__()
        self.render_mode = "human" if render else None
        self.render_enabled = render
        
        # Initialize simulation
        self.sim = Simulation(num_agents=1, render=render, rgb_array=False)

        # Action space: [vx, vy, vz]
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)

        # Observation space: [pipette_x, y, z, goal_x, y, z]
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32)

        self.max_vel = 0.05 
        self.distance_threshold = 0.001 # 1mm
        self._step_count = 0
        self.max_episode_steps = 1000

        # ✨ VISUAL GOAL MARKER SETUP
        if self.render_enabled:
            # Create a small red visual-only sphere (no physics collision)
            self.goal_marker_id = self.sim.p.createVisualShape(
                shapeType=self.sim.p.GEOM_SPHERE,
                radius=0.01, 
                rgbaColor=[1, 0, 0, 0.7] # Red and slightly transparent
            )
            # Spawn the marker body
            self.marker_body_id = self.sim.p.createMultiBody(
                baseVisualShapeIndex=self.goal_marker_id,
                basePosition=[0, 0, -1] # Hide it initially
            )

    def _get_obs(self):
        states = self.sim.get_states()
        pipette_pos = np.array(list(states.values())[0]["pipette_position"], dtype=np.float32)
        return np.concatenate([pipette_pos, self.goal.astype(np.float32)])

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.sim.reset(num_agents=1)
        
        # Initial dummy goal
        self.goal = np.zeros(3, dtype=np.float32)
        states = self.sim.get_states()
        current_pos = np.array(list(states.values())[0]["pipette_position"])

        # Generate random goal 2-10cm away
        rand_dist = self.np_random.uniform(0.02, 0.10)
        direction = self.np_random.uniform(-1, 1, size=3)
        direction /= (np.linalg.norm(direction) + 1e-8)
        
        self.goal = current_pos + direction * rand_dist
        self.goal[0] = np.clip(self.goal[0], -0.25, 0.25)
        self.goal[1] = np.clip(self.goal[1], -0.25, 0.25)
        self.goal[2] = np.clip(self.goal[2], 0.05, 0.25)

        # ✨ TELEPORT MARKER TO GOAL
        if self.render_enabled and hasattr(self, 'marker_body_id'):
            self.sim.p.resetBasePositionAndOrientation(
                self.marker_body_id, self.goal, [0, 0, 0, 1]
            )

        self._step_count = 0
        return self._get_obs(), {}

    def step(self, action):
        self._step_count += 1
        scaled_action = [[float(a * self.max_vel) for a in action] + [0.0]]
        self.sim.run(scaled_action, num_steps=1)
        
        obs = self._get_obs()
        pipette_pos = obs[:3]
        distance = float(np.linalg.norm(pipette_pos - self.goal))

        is_success = distance < self.distance_threshold
        reward = 10.0 if is_success else -distance * 10.0
        
        terminated = is_success
        truncated = self._step_count >= self.max_episode_steps
        
        return obs, float(reward), terminated, truncated, {"is_success": float(is_success), "dist_mm": distance*1000}

    def close(self):
        self.sim.close()