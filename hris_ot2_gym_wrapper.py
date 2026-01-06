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
        self.pipette_offset = [0.073, 0.0895, 0.0895]  # From CAD/URDF design
        self.robotIds = []
        self.robot_joint_info = []  # Store joint mappings per robot
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

                # 🔑 Map joint names to indices
                joint_name_to_id = {}
                for joint_index in range(p.getNumJoints(robot_id)):
                    info = p.getJointInfo(robot_id, joint_index)
                    name = info[1].decode('utf-8')
                    joint_name_to_id[name] = joint_index

                # Ensure required joints exist
                x_id = joint_name_to_id['Slider_3']
                y_id = joint_name_to_id['Slider_4']
                z_id = joint_name_to_id['Slider_5']
                self.robot_joint_info.append({
                    'x': x_id,
                    'y': y_id,
                    'z': z_id
                })
                count += 1

    def reset(self):
        for idx, rId in enumerate(self.robotIds):
            joints = self.robot_joint_info[idx]
            p.resetJointState(rId, joints['x'], 0.0)
            p.resetJointState(rId, joints['y'], 0.0)
            p.resetJointState(rId, joints['z'], 0.0)
        return self.get_states()

    def run(self, actions, num_steps=5):
        for _ in range(num_steps):
            for i, rId in enumerate(self.robotIds):
                joints = self.robot_joint_info[i]
                vx, vy, vz = actions[i]
                p.setJointMotorControl2(rId, joints['x'], p.VELOCITY_CONTROL, targetVelocity=vx, force=200)
                p.setJointMotorControl2(rId, joints['y'], p.VELOCITY_CONTROL, targetVelocity=vy, force=200)
                p.setJointMotorControl2(rId, joints['z'], p.VELOCITY_CONTROL, targetVelocity=vz, force=200)
            p.stepSimulation()
        return self.get_states()

    def get_states(self):
        states = {}
        for idx, rId in enumerate(self.robotIds):
            joints = self.robot_joint_info[idx]
            # ✅ CORRECT: Read from NAMED JOINTS
            x_pos = p.getJointState(rId, joints['x'])[0]
            y_pos = p.getJointState(rId, joints['y'])[0]
            z_pos = p.getJointState(rId, joints['z'])[0]

            pipette_pos = [
                x_pos + self.pipette_offset[0],
                y_pos + self.pipette_offset[1],
                z_pos + self.pipette_offset[2]
            ]
            states[f'r_{rId}'] = {"pipette_position": pipette_pos}
        return states

    def close(self):
        p.disconnect()


class OT2Env(gym.Env):
    def __init__(self, render=False):
        super().__init__()
        self.sim = Simulation(num_agents=1, render=render)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-2.0, high=2.0, shape=(10,), dtype=np.float32)
        # Workspace from URDF joint limits + pipette offset
        # Slider_3: [-0.18, 0.26] + 0.073 → [-0.107, 0.333]
        # Slider_4: [-0.13, 0.26] + 0.0895 → [-0.0405, 0.3495]
        # Slider_5: [0.05, 0.17] + 0.0895 → [0.1395, 0.2595]
        # BUT your original workspace seems empirically calibrated — keep it, but tighten slightly
        self.workspace_low = np.array([-0.1871, -0.1706, 0.1195], dtype=np.float32)
        self.workspace_high = np.array([0.2532, 0.2197, 0.2897], dtype=np.float32)
        self.goal_pos = np.zeros(3)
        self.steps = 0
        self.prev_dist = 0.0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.steps = 0
        self.goal_pos = self.np_random.uniform(self.workspace_low, self.workspace_high).astype(np.float32)
        state = self.sim.reset()
        curr_pos = np.array(state[next(iter(state))]['pipette_position'], dtype=np.float32)
        self.prev_dist = np.linalg.norm(curr_pos - self.goal_pos)
        return self._get_obs(curr_pos), {"goal": self.goal_pos}

    def _get_obs(self, pos):
        norm_pos = 2.0 * (pos - self.workspace_low) / (self.workspace_high - self.workspace_low) - 1.0
        norm_goal = 2.0 * (self.goal_pos - self.workspace_low) / (self.workspace_high - self.workspace_low) - 1.0
        last_action = np.zeros(3, dtype=np.float32)
        return np.concatenate([norm_pos, norm_goal, last_action, [self.prev_dist]]).astype(np.float32)

    def step(self, action):
        self.steps += 1
        max_vel = np.clip(self.prev_dist * 1.0, 0.001, 0.1)
        vel = np.clip(action, -1.0, 1.0) * max_vel

        state = self.sim.run([vel], num_steps=5)
        new_pos = np.array(state[next(iter(state))]['pipette_position'], dtype=np.float32)
        # Safety: clamp to workspace (URDF limits should prevent this, but safe guard)
        new_pos = np.clip(new_pos, self.workspace_low, self.workspace_high)
        curr_dist = np.linalg.norm(new_pos - self.goal_pos)

        reward = (self.prev_dist - curr_dist) * 500.0
        reward += 1.0 / (curr_dist + 0.01)
        reward -= 0.01

        terminated = curr_dist < 0.001  # 1 mm
        if terminated:
            reward += 100.0

        truncated = self.steps >= 1000
        self.prev_dist = curr_dist

        return self._get_obs(new_pos), float(reward), terminated, truncated, {"distance": curr_dist}