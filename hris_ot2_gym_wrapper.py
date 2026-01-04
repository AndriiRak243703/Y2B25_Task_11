import numpy as np
import gymnasium as gym
from gymnasium import spaces
import pybullet as p
import random
from sim_class import Simulation

class OT2Env(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, render=False):
        super(OT2Env, self).__init__()
        self.render = render
        self.sim = Simulation(num_agents=1, render=render)
        
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(6,), dtype=np.float32)
        
        # Workspace boundaries (in meters)
        self.workspace_low = np.array([-0.1871, -0.1706, 0.1195], dtype=np.float32)
        self.workspace_high = np.array([0.2532, 0.2197, 0.2897], dtype=np.float32)
        
        # Robot URDF Offsets
        self.base_position = np.array([0.0, 0.0, 0.03], dtype=np.float32)
        self.pipette_offset = np.array([0.073, 0.0895, 0.0895], dtype=np.float32)
        
        self.max_steps = 1000
        self.steps = 0
        self.withdrawal_timer = 0 # Track stagnation
        self.prev_dist = 0.0

    def _normalize_pos(self, pos):
        return 2.0 * (pos - self.workspace_low) / (self.workspace_high - self.workspace_low) - 1.0

    def _extract_pos(self, state_dict):
        robot_id = next(iter(state_dict))
        return np.array(state_dict[robot_id]['pipette_position'], dtype=np.float32)

    def _compute_joint_positions(self, target_pos):
        joint0 = self.base_position[0] + self.pipette_offset[0] - target_pos[0]
        joint1 = self.base_position[1] + self.pipette_offset[1] - target_pos[1]
        joint2 = target_pos[2] - self.base_position[2] - self.pipette_offset[2]
        return np.array([joint0, joint1, joint2], dtype=np.float32)

    def _validate_joint_limits(self, joints):
        JOINT_LIMITS = [(-0.25, 0.25), (-0.25, 0.25), (0.0, 0.17)]
        return np.array([np.clip(joints[i], *JOINT_LIMITS[i]) for i in range(3)], dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.steps = 0
        self.withdrawal_timer = 0
        
        # Randomize goal
        self.goal_pos = self.np_random.uniform(self.workspace_low, self.workspace_high)
        self.sim.reset(num_agents=1)
        robot_id = self.sim.robotIds[0]
        
        # Robust Start Position Validation
        for _ in range(10):
            start_pos = self.np_random.uniform(self.workspace_low, self.workspace_high)
            joints = self._compute_joint_positions(start_pos)
            valid_joints = self._validate_joint_limits(joints)
            
            # Reset joints directly in PyBullet for the new episode
            for i in range(3):
                p.resetJointState(robot_id, i, targetValue=valid_joints[i])
            
            current_pos = self._extract_pos(self.sim.get_states())
            if np.all(current_pos >= self.workspace_low) and np.all(current_pos <= self.workspace_high):
                break
        
        self.prev_dist = np.linalg.norm(current_pos - self.goal_pos)
        obs = np.concatenate([self._normalize_pos(current_pos), self._normalize_pos(self.goal_pos)]).astype(np.float32)
        return obs, {}

    def step(self, action):
        self.steps += 1
        velocity = np.clip(action, -1.0, 1.0) * 0.5 
        
        state_dict = self.sim.run([[float(velocity[0]), float(velocity[1]), float(velocity[2]), 0.0]])
        current_pos = self._extract_pos(state_dict)
        curr_dist = np.linalg.norm(current_pos - self.goal_pos)
        
        # --- ADDICTION/WITHDRAWAL REWARD ENGINE ---
        progress = self.prev_dist - curr_dist
        
        if progress > 0.0005:
            reward = progress * 5000.0  # Dopamine Hit
            self.withdrawal_timer = 0
        else:
            self.withdrawal_timer += 1
            # Exponential Withdrawal Penalty: Punishes standing still
            reward = -0.5 * (1.12 ** self.withdrawal_timer)
            reward = max(reward, -20.0) # Penalty Cap

        # Success Condition
        terminated = False
        if curr_dist < 0.001:
            reward += 1000.0 # Ultimate Jackpot
            terminated = True
        
        self.prev_dist = curr_dist
        truncated = self.steps >= self.max_steps
        
        obs = np.concatenate([self._normalize_pos(current_pos), self._normalize_pos(self.goal_pos)]).astype(np.float32)
        return obs, float(reward), terminated, truncated, {"dist": curr_dist}

    def close(self):
        self.sim.close()