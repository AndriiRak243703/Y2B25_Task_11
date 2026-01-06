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
        self.pipette_offset = [0.073, 0.0895, 0.0895]
        self.robotIds = []
        self.create_robots(num_agents)

    def create_robots(self, num_agents):
        spacing = 1
        grid_size = math.ceil(num_agents ** 0.5)
        count = 0
        for i in range(grid_size):
            for j in range(grid_size):
                if count >= num_agents: break
                pos = [-spacing * i, -spacing * j, 0.03]
                self.robotIds.append(p.loadURDF("ot_2_simulation_v6.urdf", pos, [0,0,0,1]))
                count += 1

    def reset(self):
        for rId in self.robotIds:
            for j in [0, 1, 2]: p.resetJointState(rId, j, 0)
        return self.get_states()

    def run(self, actions, num_steps=15):
        for _ in range(num_steps):
            for i, rId in enumerate(self.robotIds):
                p.setJointMotorControl2(rId, 0, p.VELOCITY_CONTROL, targetVelocity=-actions[i][0], force=500)
                p.setJointMotorControl2(rId, 1, p.VELOCITY_CONTROL, targetVelocity=-actions[i][1], force=500)
                p.setJointMotorControl2(rId, 2, p.VELOCITY_CONTROL, targetVelocity=actions[i][2], force=800)
            p.stepSimulation()
        return self.get_states()

    def get_states(self):
        states = {}
        for rId in self.robotIds:
            pos, _ = p.getBasePositionAndOrientation(rId)
            js = p.getJointStates(rId, [0, 1, 2])
            states[f'r_{rId}'] = {"pipette_position": [pos[0]-js[0][0]+0.073, pos[1]-js[1][0]+0.0895, pos[2]+js[2][0]+0.0895]}
        return states

    def close(self): p.disconnect()

class OT2Env(gym.Env):
    def __init__(self, render=False):
        super().__init__()
        self.sim = Simulation(num_agents=1, render=render)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(10,), dtype=np.float32)
        self.workspace_low = np.array([-0.1871, -0.1706, 0.1195], dtype=np.float32)
        self.workspace_high = np.array([0.2532, 0.2197, 0.2897], dtype=np.float32)
        self.goal_pos = np.zeros(3); self.steps = 0; self.prev_dist = 0.0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.steps = 0
        self.goal_pos = np.clip(self.np_random.uniform(self.workspace_low, self.workspace_high), self.workspace_low, self.workspace_high)
        state = self.sim.reset()
        curr_pos = np.array(state[next(iter(state))]['pipette_position'], dtype=np.float32)
        self.prev_dist = np.linalg.norm(curr_pos - self.goal_pos)
        return self._get_obs(curr_pos), {"goal": self.goal_pos}

    def _get_obs(self, pos):
        norm_pos = 2.0 * (pos - self.workspace_low) / (self.workspace_high - self.workspace_low) - 1.0
        norm_goal = 2.0 * (self.goal_pos - self.workspace_low) / (self.workspace_high - self.workspace_low) - 1.0
        return np.concatenate([norm_pos, norm_goal, [0,0,0], [self.prev_dist]]).astype(np.float32)

    def step(self, action):
        self.steps += 1
        # Continuous velocity scaling
        max_vel = np.clip(self.prev_dist * 1.5, 0.001, 0.25)
        vel = np.clip(action, -1, 1) * max_vel
        state = self.sim.run([vel], num_steps=15)
        new_pos = np.array(state[next(iter(state))]['pipette_position'], dtype=np.float32)
        curr_dist = np.linalg.norm(new_pos - self.goal_pos)
        
        # Reward: Progression + Sharp Gravity Well
        reward = (self.prev_dist - curr_dist) * 200.0
        reward += 1.0 / (curr_dist + 0.002) # Sharper pull
        reward -= 0.05
        
        terminated = bool(curr_dist < 0.001)
        if terminated: reward += 100.0
        self.prev_dist = curr_dist
        return self._get_obs(new_pos), float(reward), terminated, self.steps >= 1000, {"distance": curr_dist}