import numpy as np
import gymnasium as gym
from gymnasium import spaces
import pybullet as p
import pybullet_data
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import os

# ==========================================
# 🚀 SIMULATION & ENVIRONMENT (Keep Identical to Training)
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
        return self.get_state()

    def get_state(self):
        js = p.getJointStates(self.robotId, self.joints)
        pos = [js[0][0] + self.pipette_offset[0], js[1][0] + self.pipette_offset[1], js[2][0] + self.pipette_offset[2]]
        vel = [js[0][1], js[1][1], js[2][1]]
        return np.array(pos), np.array(vel)

class OT2PrecisionEnv(gym.Env):
    def __init__(self, render=False):
        super().__init__()
        self.sim = Simulation(num_agents=1, render=render)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(7,), dtype=np.float32)
        self.goal_pos = np.zeros(3)
        self.steps = 0
        self.max_vel = 0.1 

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.steps = 0
        pos, vel = self.sim.reset()
        rand_dist = self.np_random.uniform(0.01, 0.20) 
        random_dir = self.np_random.uniform(-1, 1, size=3)
        random_dir /= np.linalg.norm(random_dir) + 1e-6
        self.goal_pos = pos + (random_dir * rand_dist)
        return self._get_obs(pos, vel), {}

    def _get_obs(self, pos, vel):
        rel_pos = self.goal_pos - pos
        dist = np.linalg.norm(rel_pos)
        return np.concatenate([rel_pos, vel, [dist]]).astype(np.float32)

    def step(self, action):
        self.steps += 1
        current_pos, _ = self.sim.get_state()
        dist_to_goal = np.linalg.norm(current_pos - self.goal_pos)
        
        # Tighten brakes for fine-tuning
        slowdown_radius = 0.05
        brake_factor = 1.0
        if dist_to_goal < slowdown_radius:
            brake_factor = max(0.15, dist_to_goal / slowdown_radius) 

        scaled_action = action * self.max_vel * brake_factor
        pos, vel = self.sim.run(scaled_action, num_steps=1)
        
        new_dist = np.linalg.norm(pos - self.goal_pos)
        reward = -new_dist 
        
        if new_dist < 0.01: 
            reward += 0.5 # Increased precision incentive
            reward -= np.linalg.norm(vel) * 0.1 # Stiffer velocity penalty

        terminated = False
        if new_dist < 0.001: 
            reward += 20.0 
            terminated = True
        
        truncated = self.steps >= 1200
        info = {"dist_mm": new_dist * 1000.0, "is_success": float(terminated)}
        return self._get_obs(pos, vel), float(reward), terminated, truncated, info

# ==========================================
# 📊 FINE-TUNING LOGGER
# ==========================================
class FineTuneLogger(BaseCallback):
    def __init__(self, check_freq=5000):
        super().__init__()
        self.check_freq = check_freq
        self.success_buffer = [] 
        self.best_mm = float('inf')

    def _on_step(self) -> bool:
        if self.locals['dones'][0]:
            info = self.locals['infos'][0]
            dist_mm = info.get('dist_mm', 0.0)
            self.success_buffer.append(info.get('is_success', 0.0))
            if dist_mm < self.best_mm: self.best_mm = dist_mm

        if self.n_calls % self.check_freq == 0 and self.success_buffer:
            success_rate = np.mean(self.success_buffer[-100:]) * 100 
            print(f"STEP {self.n_calls} | Success: {success_rate:.1f}% | Best: {self.best_mm:.2f}mm")
        return True

# ==========================================
# 🏁 MAIN - FINE TUNING
# ==========================================
def main():
    print("Loading pre-trained 10M model for Fine-Tuning...")
    
    # 1. Recreate Normalized Env
    env = DummyVecEnv([lambda: OT2PrecisionEnv(render=False)])
    # Load the normalization stats from your 10M run
    env = VecNormalize.load("vec_normalize.pkl", env)
    # Stop normalization from updating further to freeze the scale
    env.training = False 
    env.norm_reward = False # Stop scaling rewards during fine-tuning

    # 2. Load Model
    # Note: We manually overwrite ent_coef and learning_rate
    model = PPO.load(
        "final_model_entropy_fix", 
        env=env,
        device="cpu", # Change to "cuda" if using GPU
        custom_objects={
            "learning_rate": 5e-5, 
            "ent_coef": 0.0001 # CRITICAL: Remove the noise
        }
    )

    print("Fine-tuning started (Low Entropy mode)...")
    callback = FineTuneLogger(check_freq=5000)
    
    # Run for 1M steps of pure stabilization
    model.learn(total_timesteps=3_000_000, callback=callback)
    
    model.save("mastered_ot2_model")
    print("Mastery model saved as mastered_ot2_model.zip")

if __name__ == "__main__":
    main()