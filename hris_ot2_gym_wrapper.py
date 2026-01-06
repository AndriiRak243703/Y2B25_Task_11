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
# Set non-interactive backend for headless docker environments
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
from clearml import Task, Logger
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv

# ==============================================================================
# 1. SIMULATION MANAGER (Optimized)
# ==============================================================================
class Simulation:
    def __init__(self, num_agents, render=True, rgb_array=False):
        self.render = render
        self.rgb_array = rgb_array
        
        # Connect to Physics Server
        mode = p.GUI if render else p.DIRECT
        self.physicsClient = p.connect(mode)
        
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -10)
        
        self.planeId = p.loadURDF("plane.urdf")
        
        # Texture logic
        self.textureId = -1
        if os.path.exists("textures"):
            texture_list = [f for f in os.listdir("textures") if f.endswith('.png')]
            if texture_list:
                random_texture = random.choice(texture_list)
                self.textureId = p.loadTexture(f"textures/{random_texture}")
        
        self.pipette_offset = [0.073, 0.0895, 0.0895]
        self.robotIds = []
        self.specimenIds = []
        
        cameraDistance = 1.1 * (math.ceil((num_agents) ** 0.3))
        p.resetDebugVisualizerCamera(cameraDistance, 90, -35, [-0.2, 0.5, 0.1])

        self.create_robots(num_agents)

    def create_robots(self, num_agents):
        spacing = 1
        grid_size = math.ceil(num_agents ** 0.5)
        agent_count = 0

        for i in range(grid_size):
            for j in range(grid_size):
                if agent_count >= num_agents: break
                position = [-spacing * i, -spacing * j, 0.03]
                
                robotId = p.loadURDF("ot_2_simulation_v6.urdf", position, [0, 0, 0, 1], flags=p.URDF_USE_INERTIA_FROM_FILE)
                
                offset = [0.18275-0.00005, 0.163-0.026, 0.057]
                spec_pos = [position[0] + offset[0], position[1] + offset[1], position[2] + offset[2]]
                planeId = p.loadURDF("custom.urdf", spec_pos, p.getQuaternionFromEuler([0, 0, -math.pi/2]))
                
                p.setCollisionFilterPair(robotId, planeId, -1, -1, enableCollision=0)
                
                if self.textureId >= 0:
                    p.changeVisualShape(planeId, -1, textureUniqueId=self.textureId)

                self.robotIds.append(robotId)
                self.specimenIds.append(planeId)
                agent_count += 1

    def reset(self):
        # Memory Leak Fix: Don't reload URDFs, just reset joints
        for robotId in self.robotIds:
            p.resetJointState(robotId, 0, 0)
            p.resetJointState(robotId, 1, 0)
            p.resetJointState(robotId, 2, 0)
        return self.get_states()

    def run(self, actions, num_steps=1):
        # TIME DILATION LOOP
        for _ in range(num_steps):
            self.apply_actions(actions)
            p.stepSimulation()
            if self.render and not self.rgb_array:
                time.sleep(1./240.)
        return self.get_states()

    def apply_actions(self, actions):
        for i, robotId in enumerate(self.robotIds):
            p.setJointMotorControl2(robotId, 0, p.VELOCITY_CONTROL, targetVelocity=-actions[i][0], force=500)
            p.setJointMotorControl2(robotId, 1, p.VELOCITY_CONTROL, targetVelocity=-actions[i][1], force=500)
            p.setJointMotorControl2(robotId, 2, p.VELOCITY_CONTROL, targetVelocity=actions[i][2], force=800)

    def get_pipette_position(self, robotId):
        robot_pos, _ = p.getBasePositionAndOrientation(robotId)
        joint_states = p.getJointStates(robotId, [0, 1, 2])
        x = robot_pos[0] - joint_states[0][0] + self.pipette_offset[0]
        y = robot_pos[1] - joint_states[1][0] + self.pipette_offset[1]
        z = robot_pos[2] + joint_states[2][0] + self.pipette_offset[2]
        return [x, y, z]

    def get_states(self):
        states = {}
        for robotId in self.robotIds:
            states[f'robotId_{robotId}'] = {"pipette_position": self.get_pipette_position(robotId)}
        return states

    def close(self):
        if p.isConnected(): p.disconnect()

# ==============================================================================
# 2. GYM ENVIRONMENT (Continuous Steps + Directional Penalty)
# ==============================================================================
class OT2Env(gym.Env):
    def __init__(self, render=False):
        super().__init__()
        self.render_mode = "human" if render else "rgb_array"
        self.sim = Simulation(num_agents=1, render=render, rgb_array=(self.render_mode == "rgb_array"))
        
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(10,), dtype=np.float32)
        
        self.workspace_low = np.array([-0.1871, -0.1706, 0.1195], dtype=np.float32)
        self.workspace_high = np.array([0.2532, 0.2197, 0.2897], dtype=np.float32)
        
        self.goal_pos = np.zeros(3)
        self.steps = 0
        self.max_steps = 1000
        self.prev_dist = 0.0
        self.prev_position = None

    def _normalize_pos(self, pos):
        return 2.0 * (pos - self.workspace_low) / (self.workspace_high - self.workspace_low) - 1.0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.steps = 0
        self.prev_position = None
        self.goal_pos = np.clip(self.np_random.uniform(self.workspace_low, self.workspace_high), self.workspace_low, self.workspace_high)
        
        state_dict = self.sim.reset()
        robot_id = self.sim.robotIds[0]
        curr_pos = np.array(state_dict[f'robotId_{robot_id}']['pipette_position'], dtype=np.float32)
        
        self.prev_dist = np.linalg.norm(curr_pos - self.goal_pos)
        self.prev_position = curr_pos.copy()
        
        return self._create_obs(curr_pos), {"goal": self.goal_pos}

    def _create_obs(self, pos):
        dist = np.linalg.norm(pos - self.goal_pos)
        max_dist = np.linalg.norm(self.workspace_high - self.workspace_low)
        norm_dist = np.clip(dist / max_dist, 0, 1)
        velocity = (pos - self.prev_position) * 240.0 if self.prev_position is not None else np.zeros(3)
        self.prev_position = pos.copy()
        
        return np.concatenate([
            self._normalize_pos(pos),
            self._normalize_pos(self.goal_pos),
            np.clip(velocity, -1, 1), 
            [norm_dist * 2 - 1]
        ]).astype(np.float32)

    def step(self, action):
        self.steps += 1
        dist_to_goal = self.prev_dist
        
        # --- "ALL THE WAY DOWN" VELOCITY SCALING ---
        # The robot slows down continuously as it gets closer.
        # At 20cm -> 0.2m/s
        # At 1mm -> 0.001m/s
        linear_vel = dist_to_goal * 1.0 
        max_vel = np.clip(linear_vel, 0.001, 0.2)
        
        velocity = np.clip(action, -1.0, 1.0) * max_vel
        
        # --- TIME DILATION ---
        # 15 physics steps per 1 Gym step. 
        # This prevents timeout when moving at 1mm/s.
        self.sim.run([[float(velocity[0]), float(velocity[1]), float(velocity[2])]], num_steps=15)
        
        state_dict = self.sim.get_states()
        new_pos = np.array(state_dict[next(iter(state_dict))]['pipette_position'], dtype=np.float32)
        curr_dist = np.linalg.norm(new_pos - self.goal_pos)
        
        # --- REWARD SHAPING ---
        # 1. Progression (Implicitly punishes moving away with negative values)
        reward = (self.prev_dist - curr_dist) * 500.0
        
        # 2. EXTRA DIRECTIONAL PENALTY (User Requested)
        # If moving AWAY from target (curr > prev), add extra pain.
        if curr_dist > self.prev_dist:
            reward -= (curr_dist - self.prev_dist) * 500.0 # Effectively doubles the penalty
            reward -= 0.5 # Flat penalty for bad decision
        
        # 3. Sharper Gravity Well 
        # Encourages being close even if stationary
        reward += 1.0 / (curr_dist + 0.005)
        
        # 4. Step Penalty (Urgency)
        reward -= 0.05
        
        terminated = False
        if curr_dist < 0.001: 
            reward += 200.0
            terminated = True
        
        self.prev_dist = curr_dist
        truncated = self.steps >= self.max_steps
        
        return self._create_obs(new_pos), float(reward), terminated, truncated, {
            'distance': float(curr_dist),
            'position': new_pos.copy()
        }