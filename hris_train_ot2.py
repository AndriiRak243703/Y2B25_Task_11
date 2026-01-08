import numpy as np
import gymnasium as gym
from gymnasium import spaces
import pybullet as p
import pybullet_data
import math
import os
import gc
import sys
import subprocess
import torch as th
from collections import deque
from clearml import Task
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, CallbackList
from stable_baselines3.common.vec_env import DummyVecEnv

# Ensure Tensorboard is installed
try:
    import tensorboard
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "tensorboard"])

# ==========================================
# 🚀 SIMULATION CLASS
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
                # Load the OT2 URDF
                robot_id = p.loadURDF("ot_2_simulation_v6.urdf", pos, [0, 0, 0, 1], useFixedBase=True)
                self.robotIds.append(robot_id)
                joint_name_to_id = {}
                for joint_index in range(p.getNumJoints(robot_id)):
                    info = p.getJointInfo(robot_id, joint_index)
                    name = info[1].decode('utf-8')
                    joint_name_to_id[name] = joint_index
                
                # Map the Slider joints
                self.robot_joint_info.append({
                    'x': joint_name_to_id['Slider_3'],
                    'y': joint_name_to_id['Slider_4'],
                    'z': joint_name_to_id['Slider_5']
                })
                count += 1

    def reset(self):
        for idx, rId in enumerate(self.robotIds):
            joints = self.robot_joint_info[idx]
            # Small random jitter to prevent overfitting to exact 0,0,0
            start_x = np.random.uniform(-0.01, 0.01)
            start_y = np.random.uniform(-0.01, 0.01)
            start_z = np.random.uniform(-0.01, 0.01)
            p.resetJointState(rId, joints['x'], start_x)
            p.resetJointState(rId, joints['y'], start_y)
            p.resetJointState(rId, joints['z'], start_z)
        return self.get_states()

    def run(self, actions, num_steps=10):
        # Run physics for multiple sub-steps to stabilize movement
        for _ in range(num_steps):
            for i, rId in enumerate(self.robotIds):
                joints = self.robot_joint_info[i]
                vx, vy, vz = actions[i]
                p.setJointMotorControl2(rId, joints['x'], p.VELOCITY_CONTROL, targetVelocity=vx, force=500)
                p.setJointMotorControl2(rId, joints['y'], p.VELOCITY_CONTROL, targetVelocity=vy, force=500)
                p.setJointMotorControl2(rId, joints['z'], p.VELOCITY_CONTROL, targetVelocity=vz, force=500)
            p.stepSimulation()
            if self.render: 
                import time
                time.sleep(1./240.)
        return self.get_states()

    def get_states(self):
        states = []
        for idx, rId in enumerate(self.robotIds):
            joints = self.robot_joint_info[idx]
            js = p.getJointStates(rId, [joints['x'], joints['y'], joints['z']])
            # Calculate Pipette Tip Position
            pos = [js[0][0] + self.pipette_offset[0], js[1][0] + self.pipette_offset[1], js[2][0] + self.pipette_offset[2]]
            vel = [js[0][1], js[1][1], js[2][1]]
            states.append((pos, vel))
        return states

    def close(self):
        p.disconnect()

# ==========================================
# 🧠 ENVIRONMENT CLASS (The Fix)
# ==========================================
class OT2ExpandingEnv(gym.Env):
    def __init__(self, render=False):
        super().__init__()
        self.sim = Simulation(num_agents=1, render=render)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
        
        # Obs: [Relative_Position(3), Velocity(3)]
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32)
        
        self.workspace_low = np.array([-0.1871, -0.1706, 0.1195], dtype=np.float32)
        self.workspace_high = np.array([0.2532, 0.2197, 0.2897], dtype=np.float32)
        
        self.goal_pos = np.zeros(3)
        self.prev_dist = 0.0
        self.steps = 0
        self.max_vel = 0.25 

        # 🎓 CURRICULUM CONFIG
        self.curriculum_radius = 0.02  # Start at 20mm
        self.max_radius = 0.30         # Max 30cm
        self.success_history = deque(maxlen=50)
        self.success_threshold_mm = 2.0 

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.steps = 0

        # 🚀 CURRICULUM LEVEL UP
        # If last 20 episodes had >80% success, expand the circle
        if len(self.success_history) >= 20:
            if np.mean(self.success_history) > 0.80 and self.curriculum_radius < self.max_radius:
                self.curriculum_radius *= 1.2
                self.curriculum_radius = min(self.curriculum_radius, self.max_radius)
                self.success_history.clear()
                print(f"🚀 EXPANDING! New Radius: {self.curriculum_radius*1000:.1f} mm")

        # 1. Reset Robot
        states = self.sim.reset()
        curr_pos = np.array(states[0][0], dtype=np.float32)

        # 2. Spawn Goal Relative to Robot
        random_dir = self.np_random.uniform(-1, 1, size=3)
        random_dir /= np.linalg.norm(random_dir) + 1e-6
        
        self.goal_pos = curr_pos + (random_dir * self.curriculum_radius)
        self.goal_pos = np.clip(self.goal_pos, self.workspace_low, self.workspace_high).astype(np.float32)
        
        # 3. Init previous distance for Delta Reward
        self.prev_dist = np.linalg.norm(curr_pos - self.goal_pos)
        
        return self._get_obs(curr_pos, np.zeros(3)), {"goal": self.goal_pos}

    def _get_obs(self, pos, vel):
        rel_pos = self.goal_pos - pos
        # Scale inputs: 1cm = 0.1 input
        return np.concatenate([rel_pos * 10.0, vel * 5.0]).astype(np.float32)

    def step(self, action):
        self.steps += 1
        action = np.clip(action, -1.0, 1.0).astype(np.float32)
        
        vel_cmd = action * self.max_vel
        states = self.sim.run([vel_cmd], num_steps=10)
        
        new_pos = np.array(states[0][0], dtype=np.float32)
        new_vel = np.array(states[0][1], dtype=np.float32)
        
        curr_dist = np.linalg.norm(new_pos - self.goal_pos)
        dist_mm = curr_dist * 1000.0
        
        # =======================
        # 🧠 DELTA REWARD LOGIC
        # =======================
        # Improvement Reward: Positive if getting closer, Negative if moving away
        reward = (self.prev_dist - curr_dist) * 100.0
        
        # Success Bonus
        terminated = False
        if dist_mm < self.success_threshold_mm:
            terminated = True
            reward += 10.0
            self.success_history.append(1)
        
        # Safety Penalty (only if hitting limits)
        if not (np.all(new_pos > self.workspace_low) and np.all(new_pos < self.workspace_high)):
            reward -= 0.5

        self.prev_dist = curr_dist # Update for next step

        truncated = False
        if self.steps >= 400:
            truncated = True
            self.success_history.append(0)
            
        return self._get_obs(new_pos, new_vel), float(reward), terminated, truncated, {
            "distance": curr_dist, 
            "radius": self.curriculum_radius
        }

# ==========================================
# 📊 LOGGING CALLBACK
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
            
            # Log custom metrics to Tensorboard/ClearML
            self.logger.record("curriculum/radius_mm", radius_mm)
            self.logger.record("curriculum/final_dist_mm", dist_mm)

        if self.n_calls % self.check_freq == 0 and self.dist_buffer:
            avg_dist = np.mean(self.dist_buffer[-50:])
            radius = self.locals['infos'][0].get('radius', 0) * 1000
            print(f"STEP {self.n_calls} | Avg Error: {avg_dist:.2f}mm | Curr Radius: {radius:.1f}mm")
        return True

# ==========================================
# 🏁 MAIN EXECUTION
# ==========================================
def main():
    # 1. ClearML Initialization
    task = Task.init(
        project_name='Mentor Group - Myrthe/Group 1', 
        task_name='Expanding_Circle_Fixed_V2',
        output_uri=True 
    )
    
    # 2. Environment Setup
    env = DummyVecEnv([lambda: OT2ExpandingEnv(render=False)])
    
    # 3. Model Setup (PPO)
    # log_std_init=-1.0 reduces initial randomness (makes it less jittery)
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=3e-4,
        batch_size=64,
        n_steps=2048,
        gamma=0.99,
        ent_coef=0.01,
        policy_kwargs=dict(log_std_init=-1.0), 
        tensorboard_log="./ppo_ot2_tensorboard/"
    )
    
    callback = CurriculumLogger()
    
    print("Starting Training (Delta Reward + Expanding Circle)...")
    try:
        model.learn(total_timesteps=1_000_000, callback=callback)
        model.save("final_model_expanding_v2")
    except KeyboardInterrupt:
        print("Training interrupted manually. Saving model...")
        model.save("interrupted_model")

if __name__ == "__main__":
    main()