import numpy as np
import gymnasium as gym
from gymnasium import spaces
import pybullet as p
import random
import os
import math
import pybullet_data
import time
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt

class Simulation:
    def __init__(self, num_agents, render=True):
        self.render = render
        mode = p.GUI if render else p.DIRECT
        self.physicsClient = p.connect(mode)
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -10)
        self.planeId = p.loadURDF("plane.urdf")
        # Pipette offset from joint origins (in meters)
        self.pipette_offset = [0.073, 0.0895, 0.0895]
        self.robotIds = []
        self.create_robots(num_agents)

    def create_robots(self, num_agents):
        spacing = 1.0
        grid_size = math.ceil(num_agents ** 0.5)
        count = 0
        for i in range(grid_size):
            for j in range(grid_size):
                if count >= num_agents:
                    break
                pos = [-spacing * i, -spacing * j, 0.03]
                robot_id = p.loadURDF("ot_2_simulation_v6.urdf", pos, [0, 0, 0, 1])
                self.robotIds.append(robot_id)
                count += 1

    def reset(self):
        for rId in self.robotIds:
            # Reset joints to zero (safe start within joint limits)
            p.resetJointState(rId, 0, 0.0)
            p.resetJointState(rId, 1, 0.0)
            p.resetJointState(rId, 2, 0.0)
        return self.get_states()

    def run(self, actions, num_steps=5):  # Reduced from 15 to 5
        for _ in range(num_steps):
            for i, rId in enumerate(self.robotIds):
                # Apply velocity control with reduced force to avoid slippage
                p.setJointMotorControl2(rId, 0, p.VELOCITY_CONTROL, targetVelocity=actions[i][0], force=200)
                p.setJointMotorControl2(rId, 1, p.VELOCITY_CONTROL, targetVelocity=actions[i][1], force=200)
                p.setJointMotorControl2(rId, 2, p.VELOCITY_CONTROL, targetVelocity=actions[i][2], force=200)
            p.stepSimulation()
        return self.get_states()

    def get_states(self):
        states = {}
        for rId in self.robotIds:
            # CORRECTED: Use JOINT POSITIONS ONLY (no base position)
            joint_states = p.getJointStates(rId, [0, 1, 2])
            x = joint_states[0][0] + self.pipette_offset[0]
            y = joint_states[1][0] + self.pipette_offset[1]
            z = joint_states[2][0] + self.pipette_offset[2]
            states[f'r_{rId}'] = {"pipette_position": [x, y, z]}
        return states

    def close(self):
        p.disconnect()


class OT2Env(gym.Env):
    def __init__(self, render=False):
        super().__init__()
        self.sim = Simulation(num_agents=1, render=render)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
        # Observation: [norm_pipette (3), norm_goal (3), action (3), prev_dist (1)] = 10
        self.observation_space = spaces.Box(low=-2.0, high=2.0, shape=(10,), dtype=np.float32)
        # Workspace limits (meters) - MUST match URDF joint limits
        self.workspace_low = np.array([-0.1871, -0.1706, 0.1195], dtype=np.float32)
        self.workspace_high = np.array([0.2532, 0.2197, 0.2897], dtype=np.float32)
        self.goal_pos = np.zeros(3)
        self.steps = 0
        self.prev_dist = 0.0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.steps = 0
        # Sample goal within workspace
        self.goal_pos = self.np_random.uniform(self.workspace_low, self.workspace_high).astype(np.float32)
        state = self.sim.reset()
        curr_pos = np.array(state[next(iter(state))]['pipette_position'], dtype=np.float32)
        # Safety clamp (should not be needed if URDF has limits, but good practice)
        curr_pos = np.clip(curr_pos, self.workspace_low, self.workspace_high)
        self.prev_dist = np.linalg.norm(curr_pos - self.goal_pos)
        return self._get_obs(curr_pos), {"goal": self.goal_pos}

    def _get_obs(self, pos):
        # Normalize position and goal to [-1, 1]
        norm_pos = 2.0 * (pos - self.workspace_low) / (self.workspace_high - self.workspace_low) - 1.0
        norm_goal = 2.0 * (self.goal_pos - self.workspace_low) / (self.workspace_high - self.workspace_low) - 1.0
        # Placeholder for last action (not used here, but keeps obs shape consistent)
        last_action = np.zeros(3, dtype=np.float32)
        return np.concatenate([norm_pos, norm_goal, last_action, [self.prev_dist]]).astype(np.float32)

    def step(self, action):
        self.steps += 1
        # Scale velocity by proximity (closer = slower)
        max_vel = np.clip(self.prev_dist * 1.0, 0.001, 0.1)  # Reduced max velocity
        vel = np.clip(action, -1.0, 1.0) * max_vel

        state = self.sim.run([vel], num_steps=5)
        new_pos = np.array(state[next(iter(state))]['pipette_position'], dtype=np.float32)
        # CRITICAL: Clamp position to workspace (defense against physics glitches)
        new_pos = np.clip(new_pos, self.workspace_low - 0.01, self.workspace_high + 0.01)
        curr_dist = np.linalg.norm(new_pos - self.goal_pos)

        # Reward design: dense progress + sharp success bonus
        progress = self.prev_dist - curr_dist
        reward = progress * 500.0  # Scaled down from 200 → 500 is reasonable
        reward += 1.0 / (curr_dist + 0.01)  # Smooth gravity well
        reward -= 0.01  # Small step penalty

        terminated = bool(curr_dist < 0.001)  # 1mm success
        if terminated:
            reward += 100.0

        truncated = self.steps >= 1000
        self.prev_dist = curr_dist

        return self._get_obs(new_pos), float(reward), terminated, truncated, {"distance": curr_dist}