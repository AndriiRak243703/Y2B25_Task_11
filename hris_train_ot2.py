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
# 🧠 ENVIRONMENT
# ==========================================
class OT2PrecisionEnv(gym.Env):
    def __init__(self, render=False):
        super().__init__()
        self.sim = Simulation(num_agents=1, render=render)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
        # Obs: Relative Pos (3) + Velocity (3) + Distance (1)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(7,), dtype=np.float32)
        
        self.goal_pos = np.zeros(3)
        self.steps = 0
        self.max_vel = 0.1 

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.steps = 0
        pos, vel = self.sim.reset()
        
        # Shotgun Reset
        rand_dist = self.np_random.uniform(0.001, 0.20) 
        random_dir = self.np_random.uniform(-1, 1, size=3)
        random_dir /= np.linalg.norm(random_dir) + 1e-6
        self.goal_pos = pos + (random_dir * rand_dist)
        
        return self._get_obs(pos, vel), {}

    def _get_obs(self, pos, vel):
        rel_pos = self.goal_pos - pos
        dist = np.linalg.norm(rel_pos)
        # REMOVED MANUAL SCALING (*100). VecNormalize handles this better.
        return np.concatenate([rel_pos, vel, [dist]]).astype(np.float32)

    def step(self, action):
        self.steps += 1
        
        # Physics & Smart Brakes
        current_pos, _ = self.sim.get_state()
        dist_to_goal = np.linalg.norm(current_pos - self.goal_pos)
        
        slowdown_radius = 0.05
        brake_factor = 1.0
        if dist_to_goal < slowdown_radius:
            brake_factor = max(0.1, dist_to_goal / slowdown_radius) 

        scaled_action = action * self.max_vel * brake_factor
        pos, vel = self.sim.run(scaled_action, num_steps=1)
        
        # 📏 Distance
        new_dist = np.linalg.norm(pos - self.goal_pos)
        dist_mm = new_dist * 1000.0
        
        # ⚖️ REWARD SYSTEM (Dense Distance Cost)
        # ----------------------------------------------------
        # 1. Distance Penalty: The closer you are, the less you lose.
        # This creates a "slope" that pulls the agent to 0 everywhere.
        reward = -new_dist 
        
        # 2. Precision Bonus (Only when very close)
        if new_dist < 0.01: 
            reward += 0.1 
            # Small penalty for high velocity near goal
            reward -= np.linalg.norm(vel) * 0.01

        # 3. Success Spike
        terminated = False
        if new_dist < 0.001: 
            reward += 10.0 # Big reward to dominate the negative sum
            terminated = True
        
        truncated = self.steps >= 1200
        
        info = {"dist_mm": dist_mm}
        return self._get_obs(pos, vel), float(reward), terminated, truncated, info

# ==========================================
# 📊 LOGGER
# ==========================================
class PerformanceLogger(BaseCallback):
    def __init__(self, check_freq=5000):
        super().__init__()
        self.check_freq = check_freq
        self.dist_buffer = []
        self.best_mm = float('inf')

    def _on_step(self) -> bool:
        # Access the underlying env for info because VecNormalize wraps it
        infos = self.locals['infos']
        for info in infos:
            if 'dist_mm' in info:
                dist_mm = info['dist_mm']
                self.dist_buffer.append(dist_mm)
                if dist_mm < self.best_mm: self.best_mm = dist_mm
                self.logger.record("train/final_dist_mm", dist_mm)

        if self.n_calls % self.check_freq == 0 and self.dist_buffer:
            avg_dist = np.mean(self.dist_buffer[-50:])
            print(f"STEP {self.n_calls} | Avg Error: {avg_dist:.2f}mm | Best: {self.best_mm:.2f}mm")
            self.dist_buffer = self.dist_buffer[-50:] 
        return True

# ==========================================
# 🏁 MAIN
# ==========================================
def main():
    task = Task.init(
        project_name='Mentor Group - Myrthe/Group 1', 
        task_name='OT2_Shotgun_Training_Normalized',
        output_uri=True 
    )
    
    # 1. Create Env
    env = DummyVecEnv([lambda: OT2PrecisionEnv(render=False)])
    
    # 2. Add VecNormalize (CRITICAL FIX)
    # This automatically scales observations and rewards to Mean=0, Std=1
    # This stabilizes PPO significantly.
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
    print("Starting Training (Normalized + Distance Cost)...")
    
    model.learn(total_timesteps=10_000_000, callback=callback)
    
    # SAVE MODEL AND NORMALIZATION STATS
    model.save("final_model_shotgun")
    env.save("vec_normalize.pkl") # Important: Save stats for loading later!

if __name__ == "__main__":
    main()