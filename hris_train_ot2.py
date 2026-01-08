import numpy as np
import gymnasium as gym
from gymnasium import spaces
import pybullet as p
import pybullet_data
import math
import os
import sys
import subprocess
import torch as th
from collections import deque
from clearml import Task
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, CallbackList
from stable_baselines3.common.vec_env import DummyVecEnv

# Ensure Tensorboard
try:
    import tensorboard
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "tensorboard"])

# ==========================================
# 🚀 SIMULATION (Standard Setup)
# ==========================================
class Simulation:
    def __init__(self, num_agents, render=False):
        self.render = render
        self.mode = p.GUI if render else p.DIRECT
        self.physicsClient = p.connect(self.mode)
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
                if count >= num_agents: break
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
            # Randomize start slightly to prevent overfitting to (0,0,0)
            start_x = np.random.uniform(-0.05, 0.05)
            start_y = np.random.uniform(-0.05, 0.05)
            start_z = np.random.uniform(-0.05, 0.05)
            p.resetJointState(rId, joints['x'], start_x)
            p.resetJointState(rId, joints['y'], start_y)
            p.resetJointState(rId, joints['z'], start_z)
        return self.get_states()

    def run(self, actions, num_steps=10):
        # Increased steps per action for smoother movement
        for _ in range(num_steps):
            for i, rId in enumerate(self.robotIds):
                joints = self.robot_joint_info[i]
                vx, vy, vz = actions[i]
                p.setJointMotorControl2(rId, joints['x'], p.VELOCITY_CONTROL, targetVelocity=vx, force=500)
                p.setJointMotorControl2(rId, joints['y'], p.VELOCITY_CONTROL, targetVelocity=vy, force=500)
                p.setJointMotorControl2(rId, joints['z'], p.VELOCITY_CONTROL, targetVelocity=vz, force=500)
            p.stepSimulation()
            if self.render: import time; time.sleep(1./240.)
        return self.get_states()

    def get_states(self):
        states = []
        for idx, rId in enumerate(self.robotIds):
            joints = self.robot_joint_info[idx]
            js = p.getJointStates(rId, [joints['x'], joints['y'], joints['z']])
            pos = [js[0][0] + self.pipette_offset[0], js[1][0] + self.pipette_offset[1], js[2][0] + self.pipette_offset[2]]
            vel = [js[0][1], js[1][1], js[2][1]]
            states.append((pos, vel))
        return states

    def close(self):
        p.disconnect()

# ==========================================
# 🧠 EXPANDING CIRCLE ENVIRONMENT
# ==========================================
class OT2ExpandingEnv(gym.Env):
    def __init__(self, render=False):
        super().__init__()
        self.sim = Simulation(num_agents=1, render=render)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
        
        # Observation: [Rel_Pos(3), Velocity(3)] -> 6 inputs (Simplified)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32)
        
        self.workspace_low = np.array([-0.1871, -0.1706, 0.1195], dtype=np.float32)
        self.workspace_high = np.array([0.2532, 0.2197, 0.2897], dtype=np.float32)
        
        self.goal_pos = np.zeros(3)
        self.steps = 0
        self.max_vel = 0.25 

        # 🎓 CURRICULUM CONFIG
        self.curriculum_radius = 0.02  # Start at 20mm (2cm) - easier than 10mm
        self.max_radius = 0.30         # End at 30cm
        self.success_history = deque(maxlen=50)
        self.success_threshold_mm = 2.0 # 2mm accuracy required

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.steps = 0

        # 🚀 LEVEL UP LOGIC
        if len(self.success_history) >= 20:
            success_rate = np.mean(self.success_history)
            # If 80% successful, expand the circle
            if success_rate > 0.80 and self.curriculum_radius < self.max_radius:
                self.curriculum_radius *= 1.1 # Increase radius by 10%
                self.curriculum_radius = min(self.curriculum_radius, self.max_radius)
                self.success_history.clear() # Reset history for new level
                print(f"🚀 EXPANDING! New Radius: {self.curriculum_radius*1000:.1f} mm")

        # 1. Reset Robot
        states = self.sim.reset()
        curr_pos = np.array(states[0][0], dtype=np.float32)

        # 2. Spawn Goal RELATIVE to Robot (Expanding Circle)
        # We pick a random direction, and place the goal exactly 'radius' away
        random_dir = self.np_random.uniform(-1, 1, size=3)
        random_dir /= np.linalg.norm(random_dir) + 1e-6 # Normalize
        
        self.goal_pos = curr_pos + (random_dir * self.curriculum_radius)
        
        # Clip goal to ensure it stays in workspace
        self.goal_pos = np.clip(self.goal_pos, self.workspace_low, self.workspace_high).astype(np.float32)
        
        return self._get_obs(curr_pos, np.zeros(3)), {"goal": self.goal_pos}

    def _get_obs(self, pos, vel):
        rel_pos = self.goal_pos - pos
        # Scale inputs so the neural network sees numbers ~1.0 instead of ~0.01
        return np.concatenate([rel_pos * 10.0, vel * 5.0]).astype(np.float32)

    def step(self, action):
        self.steps += 1
        action = np.clip(action, -1.0, 1.0).astype(np.float32)
        
        vel_cmd = action * self.max_vel
        states = self.sim.run([vel_cmd], num_steps=10) # Run physics
        
        new_pos = np.array(states[0][0], dtype=np.float32)
        new_vel = np.array(states[0][1], dtype=np.float32)
        
        # Calculate Distance
        dist = np.linalg.norm(new_pos - self.goal_pos)
        dist_mm = dist * 1000.0
        
        # =======================
        # 🧠 REWARD FUNCTION
        # =======================
        # 1. Distance Penalty (Continuous)
        # We want to minimize distance. 
        reward = -dist 

        # 2. Success Bonus
        terminated = False
        if dist_mm < self.success_threshold_mm:
            terminated = True
            reward += 10.0 # Big bonus for reaching goal
            self.success_history.append(1)
        
        # 3. Time Penalty (encourage speed)
        reward -= 0.05

        # 4. Workspace Penalty (Don't hit walls)
        # (This is the "loose" leash - only penalize if hitting limits)
        if not (np.all(new_pos > self.workspace_low) and np.all(new_pos < self.workspace_high)):
            reward -= 1.0

        # =======================
        # 🛑 TERMINATION
        # =======================
        truncated = False
        if self.steps >= 400:
            truncated = True
            self.success_history.append(0) # Failed this episode
            
        return self._get_obs(new_pos, new_vel), float(reward), terminated, truncated, {
            "distance": dist, 
            "radius": self.curriculum_radius
        }

# ==========================================
# 🧠 TRAINING CALLBACK
# ==========================================
class CurriculumLogger(BaseCallback):
    def __init__(self, check_freq=2048):
        super().__init__()
        self.check_freq = check_freq
        self.dist_buffer = []

    def _on_step(self) -> bool:
        if self.locals['dones'][0]:
            info = self.locals['infos'][0]
            dist_mm = info.get('distance', 1.0) * 1000
            radius_mm = info.get('radius', 0.0) * 1000
            self.dist_buffer.append(dist_mm)
            
            self.logger.record("curriculum/radius_mm", radius_mm)
            self.logger.record("curriculum/final_dist_mm", dist_mm)

        if self.n_calls % self.check_freq == 0 and self.dist_buffer:
            avg_dist = np.mean(self.dist_buffer[-50:])
            radius = self.locals['infos'][0].get('radius', 0) * 1000
            print(f"STEP {self.n_calls} | Avg Error: {avg_dist:.2f}mm | Curr Radius: {radius:.1f}mm")
        return True

def main():
    # Setup ClearML
    task = Task.init(
        project_name='Mentor Group - Myrthe/Group 1', 
        task_name='Expanding_Circle_Fixed',
        output_uri=True 
    )
    
    env = DummyVecEnv([lambda: OT2ExpandingEnv(render=False)])
    
    # Initialize PPO with a smaller standard deviation (log_std_init=-1)
    # This helps it not jitter too much at the start
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=3e-4,
        batch_size=64,
        n_steps=2048,
        ent_coef=0.01,
        policy_kwargs=dict(log_std_init=-1.0),
        tensorboard_log="./ppo_ot2_tensorboard/"
    )
    
    callback = CurriculumLogger()
    
    print("Starting Training (Expanding Circle)...")
    model.learn(total_timesteps=1_000_000, callback=callback)
    model.save("final_model_expanding")

if __name__ == "__main__":
    main()