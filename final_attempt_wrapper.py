import numpy as np
import gymnasium as gym
from gymnasium import spaces
import pybullet as p
import math
import pybullet_data

class Simulation:
    def __init__(self, num_agents, render=True):
        self.render = render
        mode = p.GUI if render else p.DIRECT
        self.physicsClient = p.connect(mode)
        
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        p.configureDebugVisualizer(p.COV_ENABLE_SEGMENTATION_MARK_PREVIEW, 0)
        p.configureDebugVisualizer(p.COV_ENABLE_DEPTH_BUFFER_PREVIEW, 0)
        p.configureDebugVisualizer(p.COV_ENABLE_RGB_BUFFER_PREVIEW, 0)
        
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -10)
        
        self.planeId = p.loadURDF("plane.urdf")
        self.pipette_offset = [0.073, 0.0895, 0.0895]
        
        self.robotIds = []
        self.robot_joint_info = []
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
                robot_id = p.loadURDF("ot_2_simulation_v6.urdf", pos, [0, 0, 0, 1], useFixedBase=True)
                self.robotIds.append(robot_id)

                joint_name_to_id = {}
                for joint_index in range(p.getNumJoints(robot_id)):
                    info = p.getJointInfo(robot_id, joint_index)
                    name = info[1].decode('utf-8')
                    joint_name_to_id[name] = joint_index

                self.robot_joint_info.append({
                    'x': joint_name_to_id['Slider_3'],
                    'y': joint_name_to_id['Slider_4'],
                    'z': joint_name_to_id['Slider_5']
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
                p.setJointMotorControl2(rId, joints['x'], p.VELOCITY_CONTROL, targetVelocity=vx, force=500)
                p.setJointMotorControl2(rId, joints['y'], p.VELOCITY_CONTROL, targetVelocity=vy, force=500)
                p.setJointMotorControl2(rId, joints['z'], p.VELOCITY_CONTROL, targetVelocity=vz, force=800)
            p.stepSimulation()
        return self.get_states()

    def get_states(self):
        states = {}
        for idx, rId in enumerate(self.robotIds):
            joints = self.robot_joint_info[idx]
            js = p.getJointStates(rId, [joints['x'], joints['y'], joints['z']])
            
            x_pos, x_vel = js[0][0], js[0][1]
            y_pos, y_vel = js[1][0], js[1][1]
            z_pos, z_vel = js[2][0], js[2][1]

            pipette_pos = [
                x_pos + self.pipette_offset[0],
                y_pos + self.pipette_offset[1],
                z_pos + self.pipette_offset[2]
            ]
            pipette_vel = [x_vel, y_vel, z_vel]

            states[f'r_{rId}'] = {
                "pipette_position": pipette_pos,
                "pipette_velocity": pipette_vel
            }
        return states

    def close(self):
        p.disconnect()


class OT2Env(gym.Env):
    def __init__(self, render=False):
        super().__init__()
        self.sim = Simulation(num_agents=1, render=render)
        
        # Action: [v_x, v_y, v_z]
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
        
        # Observation: [norm_pos(3), norm_goal(3), norm_vel(3), last_action(3), prev_dist(1)] = 13
        self.observation_space = spaces.Box(low=-2.0, high=2.0, shape=(13,), dtype=np.float32)
        
        self.workspace_low = np.array([-0.1871, -0.1706, 0.1195], dtype=np.float32)
        self.workspace_high = np.array([0.2532, 0.2197, 0.2897], dtype=np.float32)
        
        self.goal_pos = np.zeros(3)
        self.steps = 0
        self.prev_dist = 0.0
        self.last_action = np.zeros(3, dtype=np.float32)

        # For velocity normalization
        self.max_vel = 0.1  # m/s — matches your scaled action cap

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.steps = 0
        self.last_action = np.zeros(3, dtype=np.float32)
        
        self.goal_pos = self.np_random.uniform(self.workspace_low, self.workspace_high).astype(np.float32)
        
        state = self.sim.reset()
        curr_pos = np.array(state[next(iter(state))]['pipette_position'], dtype=np.float32)
        curr_pos = np.clip(curr_pos, self.workspace_low, self.workspace_high)
        
        self.prev_dist = np.linalg.norm(curr_pos - self.goal_pos)
        
        return self._get_obs(curr_pos, np.zeros(3)), {"goal": self.goal_pos}

    def _get_obs(self, pos, vel):
        norm_pos = 2.0 * (pos - self.workspace_low) / (self.workspace_high - self.workspace_low) - 1.0
        norm_goal = 2.0 * (self.goal_pos - self.workspace_low) / (self.workspace_high - self.workspace_low) - 1.0
        norm_vel = vel / self.max_vel  # [-1, 1] approx
        return np.concatenate([norm_pos, norm_goal, norm_vel, self.last_action, [self.prev_dist]]).astype(np.float32)

    def step(self, action):
        self.steps += 1
        self.last_action = np.clip(action, -1.0, 1.0).astype(np.float32)
        
        # Aggressive velocity scaling near goal
        max_vel = min(0.1, 0.3 * self.prev_dist)  # Slower near target
        vel_cmd = self.last_action * max_vel

        state = self.sim.run([vel_cmd], num_steps=5)
        new_pos = np.array(state[next(iter(state))]['pipette_position'], dtype=np.float32)
        new_vel = np.array(state[next(iter(state))]['pipette_velocity'], dtype=np.float32)
        
        new_pos = np.clip(new_pos, self.workspace_low - 0.01, self.workspace_high + 0.01)
        curr_dist = np.linalg.norm(new_pos - self.goal_pos)

        # --- Improved Reward ---
        progress = (self.prev_dist - curr_dist) * 500.0
        proximity = 1.0 / (curr_dist + 0.01)
        time_penalty = -0.01
        action_penalty = -0.1 * np.linalg.norm(self.last_action)
        vel_penalty = -0.05 * np.linalg.norm(new_vel)

        reward = progress + proximity + time_penalty + action_penalty + vel_penalty

        # --- Success condition ---
        terminated = bool(curr_dist < 0.001)
        if terminated:
            reward += 100.0

        truncated = self.steps >= 1000
        self.prev_dist = curr_dist

        return self._get_obs(new_pos, new_vel), float(reward), terminated, truncated, {"distance": curr_dist}