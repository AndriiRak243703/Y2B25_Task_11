import numpy as np
import gymnasium as gym
from gymnasium import spaces
import pybullet as p
import math
import pybullet_data

class Simulation:
    def __init__(self, num_agents, render=True):
        self.render = render
        # DIRECT mode is much faster for headless training
        mode = p.GUI if render else p.DIRECT
        self.physicsClient = p.connect(mode)
        
        # 🚀 FPS Optimization: Disable all PyBullet visual debugging overhead
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        p.configureDebugVisualizer(p.COV_ENABLE_SEGMENTATION_MARK_PREVIEW, 0)
        p.configureDebugVisualizer(p.COV_ENABLE_DEPTH_BUFFER_PREVIEW, 0)
        p.configureDebugVisualizer(p.COV_ENABLE_RGB_BUFFER_PREVIEW, 0)
        
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -10)
        
        self.planeId = p.loadURDF("plane.urdf")
        
        # Pipette offset from joint origins (Matches your code)
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
                
                # Load URDF with fixed base for stability
                # Ensure 'ot_2_simulation_v6.urdf' is in the same folder
                robot_id = p.loadURDF("ot_2_simulation_v6.urdf", pos, [0, 0, 0, 1], useFixedBase=True)
                self.robotIds.append(robot_id)

                # 🔑 Dynamic Joint Mapping
                # This ensures we find Slider_3/4/5 even if URDF order changes
                joint_name_to_id = {}
                for joint_index in range(p.getNumJoints(robot_id)):
                    info = p.getJointInfo(robot_id, joint_index)
                    name = info[1].decode('utf-8')
                    joint_name_to_id[name] = joint_index

                # Store the indices for the simulation loop
                self.robot_joint_info.append({
                    'x': joint_name_to_id['Slider_3'],
                    'y': joint_name_to_id['Slider_4'],
                    'z': joint_name_to_id['Slider_5']
                })
                count += 1

    def reset(self):
        # Reset all robots to zero position
        for idx, rId in enumerate(self.robotIds):
            joints = self.robot_joint_info[idx]
            p.resetJointState(rId, joints['x'], 0.0)
            p.resetJointState(rId, joints['y'], 0.0)
            p.resetJointState(rId, joints['z'], 0.0)
        return self.get_states()

    def run(self, actions, num_steps=5):
        # 🚀 FPS Optimization: Minimize the Python-C++ loop overhead
        for _ in range(num_steps):
            for i, rId in enumerate(self.robotIds):
                joints = self.robot_joint_info[i]
                vx, vy, vz = actions[i]
                
                # Apply velocity control (Force increased to 500/800 to match your config)
                # Note: We use direct velocity. PPO will learn to output negative values
                # if the axis is inverted, so we don't need to hardcode the negative sign here.
                p.setJointMotorControl2(rId, joints['x'], p.VELOCITY_CONTROL, targetVelocity=vx, force=500)
                p.setJointMotorControl2(rId, joints['y'], p.VELOCITY_CONTROL, targetVelocity=vy, force=500)
                p.setJointMotorControl2(rId, joints['z'], p.VELOCITY_CONTROL, targetVelocity=vz, force=800)
            
            p.stepSimulation()
        
        return self.get_states()

    def get_states(self):
        states = {}
        for idx, rId in enumerate(self.robotIds):
            joints = self.robot_joint_info[idx]
            
            # 🚀 FPS Optimization: Get all 3 joint states in a single call
            js = p.getJointStates(rId, [joints['x'], joints['y'], joints['z']])
            
            # Extract positions (index 0 of the tuple for each joint)
            x_pos = js[0][0]
            y_pos = js[1][0]
            z_pos = js[2][0]

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
        
        # Action: [v_x, v_y, v_z]
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
        
        # Observation: [norm_pipette (3), norm_goal (3), action (3), prev_dist (1)] = 10
        self.observation_space = spaces.Box(low=-2.0, high=2.0, shape=(10,), dtype=np.float32)
        
        # Workspace limits (Based on your URDF limits + offsets)
        self.workspace_low = np.array([-0.1871, -0.1706, 0.1195], dtype=np.float32)
        self.workspace_high = np.array([0.2532, 0.2197, 0.2897], dtype=np.float32)
        
        self.goal_pos = np.zeros(3)
        self.steps = 0
        self.prev_dist = 0.0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.steps = 0
        
        # Sample random goal inside workspace
        self.goal_pos = self.np_random.uniform(self.workspace_low, self.workspace_high).astype(np.float32)
        
        state = self.sim.reset()
        curr_pos = np.array(state[next(iter(state))]['pipette_position'], dtype=np.float32)
        
        # Clamp initial position for safety
        curr_pos = np.clip(curr_pos, self.workspace_low, self.workspace_high)
        
        self.prev_dist = np.linalg.norm(curr_pos - self.goal_pos)
        
        return self._get_obs(curr_pos), {"goal": self.goal_pos}

    def _get_obs(self, pos):
        # Normalize position and goal to [-1, 1] range for PPO stability
        norm_pos = 2.0 * (pos - self.workspace_low) / (self.workspace_high - self.workspace_low) - 1.0
        norm_goal = 2.0 * (self.goal_pos - self.workspace_low) / (self.workspace_high - self.workspace_low) - 1.0
        
        # Placeholder for last action
        last_action = np.zeros(3, dtype=np.float32)
        
        return np.concatenate([norm_pos, norm_goal, last_action, [self.prev_dist]]).astype(np.float32)

    def step(self, action):
        self.steps += 1
        
        # Dynamic Velocity Scaling:
        # Move faster when far, move slower when close to avoid overshooting
        max_vel = np.clip(self.prev_dist * 1.0, 0.001, 0.1)
        vel = np.clip(action, -1.0, 1.0) * max_vel

        state = self.sim.run([vel], num_steps=5)
        new_pos = np.array(state[next(iter(state))]['pipette_position'], dtype=np.float32)
        
        # Safety clamp
        new_pos = np.clip(new_pos, self.workspace_low - 0.01, self.workspace_high + 0.01)
        
        curr_dist = np.linalg.norm(new_pos - self.goal_pos)

        # --- Reward Engineering ---
        # 1. Dense progress reward
        reward = (self.prev_dist - curr_dist) * 500.0
        
        # 2. Gravity well
        reward += 1.0 / (curr_dist + 0.01)
        
        # 3. Small time penalty
        reward -= 0.01

        # 4. Success bonus (1mm accuracy)
        terminated = bool(curr_dist < 0.001)
        if terminated:
            reward += 100.0

        truncated = self.steps >= 1000
        self.prev_dist = curr_dist

        return self._get_obs(new_pos), float(reward), terminated, truncated, {"distance": curr_dist}