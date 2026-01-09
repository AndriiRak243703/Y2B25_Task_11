import numpy as np
import gymnasium as gym
from gymnasium import spaces
import pybullet as p
import pybullet_data
import sys
import subprocess
import os
from typing import Callable
from clearml import Task
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

# Ensure Tensorboard
try:
    import tensorboard
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "tensorboard"])

# ==========================================
# 📉 LEARNING RATE SCHEDULER
# ==========================================
def linear_schedule(initial_value: float) -> Callable[[float], float]:
    def func(progress_remaining: float) -> float:
        return progress_remaining * initial_value
    return func

# ==========================================
# 🚀 SIMULATION (240Hz)
# ==========================================
class Simulation:
    def __init__(self, num_agents, render=False):
        self.render = render
        self.mode = p.GUI if render else p.DIRECT
        self.physicsClient = p.connect(self.mode)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, 0)
        self.planeId = p.loadURDF("plane.urdf")
        self.pipette_offset = [0.073, 0.0895, 0.0895]
        self.robotId = None
        self.create_robot()

    def create_robot(self):
        self.robotId = p.loadURDF("ot_2_simulation_v6.urdf", [0,0,0], useFixedBase=True)
        self.joints = [0, 1, 2] 

    def reset(self):
        start_pos = np.random.uniform(-0.15, 0.15, 3) 
        for i, joint in enumerate(self.joints):
            p.resetJointState(self.robotId, joint, start_pos[i])
        return self.get_state()

    def run(self, action, num_steps=1): 
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
        pos = [js[0][0] + self.pipette_offset[0], js[1][0] + self.pipette_offset[1], js[2][0] + self.pipette_offset[2]]
        vel = [js[0][1], js[1][1], js[2][1]]
        return np.array(pos), np.array(vel)

# ==========================================
# 🧠 ENVIRONMENT (Tolerance Curriculum)
# ==========================================
class OT2PrecisionEnv(gym.Env):
    def __init__(self, render=False):
        super().__init__()
        self.sim = Simulation(num_agents=1, render=render)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(7,), dtype=np.float32)
        
        self.goal_pos = np.zeros(3)
        self.steps = 0
        self.max_vel = 0.1 
        
        # 🎯 TOLERANCE CURRICULUM
        # We start by accepting 50mm (0.05) errors.
        # As the agent succeeds, we shrink this to 1mm (0.001).
        self.success_count = 0
        self.current_threshold = 0.05 

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.steps = 0
        pos, vel = self.sim.reset()
        
        # 🎲 FULL WORKSPACE START (As requested)
        # Always spawn between 10mm and 200mm away
        rand_dist = self.np_random.uniform(0.01, 0.20) 
        
        random_dir = self.np_random.uniform(-1, 1, size=3)
        random_dir /= np.linalg.norm(random_dir) + 1e-6
        self.goal_pos = pos + (random_dir * rand_dist)
        
        # Update Threshold based on Total Successes
        # If we have > 200 successes, tighten to 1cm
        # If we have > 500 successes, tighten to 1mm
        if self.success_count > 1000:
            self.current_threshold = 0.001 # 1mm (Mastery)
        elif self.success_count > 500:
            self.current_threshold = 0.01 # 1cm (Intermediate)
        else:
            self.current_threshold = 0.05 # 5cm (Beginner)

        return self._get_obs(pos, vel), {}

    def _get_obs(self, pos, vel):
        rel_pos = self.goal_pos - pos
        dist = np.linalg.norm(rel_pos)
        return np.concatenate([rel_pos, vel, [dist]]).astype(np.float32)

    def step(self, action):
        self.steps += 1
        
        # Physics
        current_pos, _ = self.sim.get_state()
        dist_to_goal = np.linalg.norm(current_pos - self.goal_pos)
        
        # Smart Brakes
        slowdown_radius = 0.05
        brake_factor = 1.0
        if dist_to_goal < slowdown_radius:
            brake_factor = max(0.2, dist_to_goal / slowdown_radius) 

        scaled_action = action * self.max_vel * brake_factor
        pos, vel = self.sim.run(scaled_action, num_steps=1)
        
        # 📏 Distance
        new_dist = np.linalg.norm(pos - self.goal_pos)
        dist_mm = new_dist * 1000.0
        
        # ⚖️ REWARD SYSTEM
        # 1. Constant Pull to Center
        reward = -new_dist 
        
        # 2. Precision Zone (1cm) - Always Active to encourage fine movement
        if new_dist < 0.01: 
            reward += 0.1 
            reward -= np.linalg.norm(vel) * 0.05

        # 3. Dynamic Success Threshold
        terminated = False
        if new_dist < self.current_threshold: 
            reward += 10.0 
            terminated = True
            self.success_count += 1 # Count success to shrink target later
        
        truncated = self.steps >= 1200
        
        info = {"dist_mm": dist_mm, "threshold": self.current_threshold}
        return self._get_obs(pos, vel), float(reward), terminated, truncated, info

# ==========================================
# 📊 LOGGER (FIXED)
# ==========================================
class PerformanceLogger(BaseCallback):
    def __init__(self, check_freq=5000):
        super().__init__()
        self.check_freq = check_freq
        self.dist_buffer = []
        self.success_buffer = [] 
        self.best_mm = float('inf')

    def _on_step(self) -> bool:
        # BUG FIX: Only log success/dist when the episode actually ENDS
        if self.locals['dones'][0]:
            info = self.locals['infos'][0]
            
            dist_mm = info.get('dist_mm', 0.0)
            threshold = info.get('threshold', 0.05)
            
            # If distance < threshold, it was a success
            is_success = 1.0 if dist_mm < (threshold * 1000) + 0.1 else 0.0
            
            self.dist_buffer.append(dist_mm)
            self.success_buffer.append(is_success)
            
            if dist_mm < self.best_mm: self.best_mm = dist_mm

        if self.n_calls % self.check_freq == 0 and self.dist_buffer:
            avg_dist = np.mean(self.dist_buffer[-50:])
            success_rate = np.mean(self.success_buffer[-100:]) * 100 
            
            print(f"STEP {self.n_calls} | Success: {success_rate:.1f}% | Avg Error: {avg_dist:.2f}mm | Best: {self.best_mm:.2f}mm")
            
            self.logger.record("train/success_rate", success_rate)
            self.logger.record("train/avg_error_mm", avg_dist)
            
            self.dist_buffer = self.dist_buffer[-50:] 
            self.success_buffer = self.success_buffer[-100:]
        return True

# ==========================================
# 🏁 MAIN
# ==========================================
def main():
    task = Task.init(
        project_name='Mentor Group - Myrthe/Group 1', 
        task_name='OT2_Tolerance_Curriculum',
        output_uri=True 
    )
    
    env = DummyVecEnv([lambda: OT2PrecisionEnv(render=False)])
    
    # Normalize inputs/rewards for stability
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.)
    
    model = PPO(
        "MlpPolicy", 
        env, 
        verbose=1, 
        learning_rate=linear_schedule(3e-4),
        n_steps=4096,           
        batch_size=64,
        ent_coef=0.0,           
        gae_lambda=0.95,
        clip_range=0.2,
        tensorboard_log="./ppo_logs/"
    )
    
    callback = PerformanceLogger(check_freq=5000)
    print("Starting Training (Tolerance Curriculum - Start Far, Aim Big)...")
    
    model.learn(total_timesteps=10_000_000, callback=callback)
    
    model.save("final_model_tolerance")
    env.save("vec_normalize.pkl") 

if __name__ == "__main__":
    main()