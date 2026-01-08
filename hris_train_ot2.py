import numpy as np
import gymnasium as gym
from gymnasium import spaces
import pybullet as p
import pybullet_data
import math
import sys
import subprocess
import os
from collections import deque
from clearml import Task
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv

# Ensure Tensorboard
try:
    import tensorboard
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "tensorboard"])

# ==========================================
# 🚀 SIMULATION (Optimized for Speed)
# ==========================================
class Simulation:
    def __init__(self, num_agents, render=False):
        self.render = render
        self.mode = p.GUI if render else p.DIRECT
        self.physicsClient = p.connect(self.mode)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, 0) # Zero gravity helps simplified movement learning
        self.planeId = p.loadURDF("plane.urdf")
        self.pipette_offset = [0.073, 0.0895, 0.0895]
        self.robotId = None
        self.create_robot()

    def create_robot(self):
        # Spawn 1 robot at origin
        self.robotId = p.loadURDF("ot_2_simulation_v6.urdf", [0,0,0], useFixedBase=True)
        # 0,1,2 are the slider joints in this URDF
        self.joints = [0, 1, 2] 

    def reset(self):
        # Jitter start position to prevent overfitting
        start_pos = np.random.uniform(-0.01, 0.01, 3)
        for i, joint in enumerate(self.joints):
            p.resetJointState(self.robotId, joint, start_pos[i])
        return self.get_state()

    def run(self, action, num_steps=5):
        vx, vy, vz = action
        p.setJointMotorControl2(self.robotId, 0, p.VELOCITY_CONTROL, targetVelocity=vx, force=1000)
        p.setJointMotorControl2(self.robotId, 1, p.VELOCITY_CONTROL, targetVelocity=vy, force=1000)
        p.setJointMotorControl2(self.robotId, 2, p.VELOCITY_CONTROL, targetVelocity=vz, force=1000)
        
        for _ in range(num_steps):
            p.stepSimulation()
            if self.render: 
                import time
                time.sleep(1./240.)
        return self.get_state()

    def get_state(self):
        js = p.getJointStates(self.robotId, self.joints)
        # Position + Pipette Offset
        pos = [js[0][0] + self.pipette_offset[0], 
               js[1][0] + self.pipette_offset[1], 
               js[2][0] + self.pipette_offset[2]]
        vel = [js[0][1], js[1][1], js[2][1]]
        return np.array(pos), np.array(vel)

# ==========================================
# 🧠 ENVIRONMENT (Directional Reward)
# ==========================================
class OT2DirectionalEnv(gym.Env):
    def __init__(self, render=False):
        super().__init__()
        self.sim = Simulation(num_agents=1, render=render)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
        # Obs: Relative Pos (3) + Velocity (3)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32)
        
        self.goal_pos = np.zeros(3)
        self.steps = 0
        self.max_vel = 0.5 
        
        # 🎓 CURRICULUM
        self.curriculum_radius = 0.05 # Start at 50mm
        self.max_radius = 0.30        # Max 300mm
        self.success_history = deque(maxlen=50)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.steps = 0

        # Auto-Expand Curriculum
        if len(self.success_history) >= 20 and np.mean(self.success_history) > 0.8:
            if self.curriculum_radius < self.max_radius:
                self.curriculum_radius = min(self.curriculum_radius * 1.2, self.max_radius)
                self.success_history.clear()
                print(f"🚀 EXPANDING! Radius: {self.curriculum_radius*1000:.1f} mm")

        pos, vel = self.sim.reset()
        
        # Spawn Goal
        random_dir = self.np_random.uniform(-1, 1, size=3)
        random_dir /= np.linalg.norm(random_dir) + 1e-6
        self.goal_pos = pos + (random_dir * self.curriculum_radius)
        
        return self._get_obs(pos, vel), {}

    def _get_obs(self, pos, vel):
        rel_pos = self.goal_pos - pos
        # Scale inputs so 1cm = 1.0 (High visibility for network)
        return np.concatenate([rel_pos * 100.0, vel * 10.0]).astype(np.float32)

    def step(self, action):
        self.steps += 1
        action = np.clip(action, -1.0, 1.0).astype(np.float32)
        
        # Physics Step
        pos, vel = self.sim.run(action * self.max_vel)
        
        # Metrics
        dist = np.linalg.norm(pos - self.goal_pos)
        dist_mm = dist * 1000.0
        
        # =======================
        # 🧠 REWARD LOGIC
        # =======================
        # 1. Distance Funnel (Standard)
        reward = -dist * 10.0 
        
        # 2. Directional Alignment (The "Homing Beacon")
        vec_to_goal = self.goal_pos - pos
        vec_to_goal /= (np.linalg.norm(vec_to_goal) + 1e-6)
        alignment = np.dot(action, vec_to_goal)
        reward += alignment * 1.0 
        
        # 3. Success Bonus
        terminated = False
        if dist < 0.005: # 5mm
            reward += 20.0
            terminated = True
            self.success_history.append(1)
        
        truncated = self.steps >= 200
        if truncated: self.success_history.append(0)

        info = {
            "dist_mm": dist_mm, 
            "radius_mm": self.curriculum_radius * 1000
        }
        return self._get_obs(pos, vel), float(reward), terminated, truncated, info

# ==========================================
# 📊 CUSTOM LOGGER (Avg & Best)
# ==========================================
class PerformanceLogger(BaseCallback):
    def __init__(self, check_freq=5000):
        super().__init__()
        self.check_freq = check_freq
        self.dist_buffer = []
        self.best_mm = float('inf')

    def _on_step(self) -> bool:
        if self.locals['dones'][0]:
            info = self.locals['infos'][0]
            dist_mm = info.get('dist_mm', 0.0)
            
            self.dist_buffer.append(dist_mm)
            if dist_mm < self.best_mm:
                self.best_mm = dist_mm
            
            # Log to ClearML/Tensorboard
            self.logger.record("curriculum/radius_mm", info.get('radius_mm'))
            self.logger.record("curriculum/final_dist_mm", dist_mm)

        if self.n_calls % self.check_freq == 0 and self.dist_buffer:
            avg_dist = np.mean(self.dist_buffer[-100:]) # Avg of last 100 episodes
            radius = self.locals['infos'][0].get('radius_mm', 0)
            
            print(f"STEP {self.n_calls} | Radius: {radius:.1f}mm | Avg Error: {avg_dist:.2f}mm | Best Ever: {self.best_mm:.2f}mm")
            
            # Reset buffer slightly to keep Average fresh
            self.dist_buffer = self.dist_buffer[-100:] 
            
        return True

# ==========================================
# 🏁 MAIN
# ==========================================
def main():
    # 1. ClearML Init (Correct Project Name)
    task = Task.init(
        project_name='Mentor Group - Myrthe/Group 1', 
        task_name='OT2_Directional_Optimized',
        output_uri=True 
    )
    
    # 2. Setup Env & Model
    env = DummyVecEnv([lambda: OT2DirectionalEnv(render=False)])
    
    # High Learning Rate (1e-3) because the Reward is very clear
    model = PPO(
        "MlpPolicy", 
        env, 
        verbose=1, 
        learning_rate=1e-3, 
        ent_coef=0.01,
        tensorboard_log="./ppo_logs/"
    )
    
    # 3. Train with Logger
    callback = PerformanceLogger(check_freq=5000)
    print("Starting Training (Directional Reward + Full Logging)...")
    
    try:
        model.learn(total_timesteps=1_000_000, callback=callback)
        model.save("final_model_directional")
    except KeyboardInterrupt:
        model.save("interrupted_model")
        print("Saved interrupt model.")

if __name__ == "__main__":
    main()