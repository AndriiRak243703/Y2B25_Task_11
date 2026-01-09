import numpy as np
import gymnasium as gym
from gymnasium import spaces
import pybullet as p
import pybullet_data
import sys
import subprocess
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
# 🚀 SIMULATION (High Frequency Update)
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
        start_pos = np.random.uniform(-0.1, 0.1, 3)
        for i, joint in enumerate(self.joints):
            p.resetJointState(self.robotId, joint, start_pos[i])
        return self.get_state()

    def run(self, action, num_steps=1): 
        # CHANGED: num_steps=1 for 240Hz control (crucial for <1mm precision)
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
# 🧠 ENVIRONMENT (Precision Mode)
# ==========================================
class OT2PrecisionEnv(gym.Env):
    def __init__(self, render=False):
        super().__init__()
        self.sim = Simulation(num_agents=1, render=render)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
        # Observation: Relative Pos (3) + Velocity (3) + Distance Scalar (1)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(7,), dtype=np.float32)
        
        self.goal_pos = np.zeros(3)
        self.steps = 0
        self.max_vel = 0.1 # 10 cm/s limit
        self.prev_dist = 0.0
        
        # Curriculum: Start small to ensure it learns to STOP first
        self.curriculum_radius = 0.05 
        self.max_radius = 0.20
        self.success_history = deque(maxlen=20)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.steps = 0

        # Curriculum expansion
        if len(self.success_history) >= 20 and np.mean(self.success_history) > 0.8:
            self.curriculum_radius = min(self.curriculum_radius * 1.2, self.max_radius)
            self.success_history.clear()
            print(f"🚀 EXPANDING! Radius: {self.curriculum_radius*1000:.1f} mm")

        pos, vel = self.sim.reset()
        
        random_dir = self.np_random.uniform(-1, 1, size=3)
        random_dir /= np.linalg.norm(random_dir) + 1e-6
        self.goal_pos = pos + (random_dir * self.curriculum_radius)
        
        self.prev_dist = np.linalg.norm(pos - self.goal_pos)
        return self._get_obs(pos, vel), {}

    def _get_obs(self, pos, vel):
        rel_pos = self.goal_pos - pos
        dist = np.linalg.norm(rel_pos)
        # We include the raw distance as a feature
        return np.concatenate([rel_pos * 100.0, vel * 10.0, [dist * 100.0]]).astype(np.float32)

    def step(self, action):
        self.steps += 1
        
        current_pos, _ = self.sim.get_state()
        dist_to_goal = np.linalg.norm(current_pos - self.goal_pos)
        
        # 🛑 Micro-Braking Logic
        slowdown_radius = 0.05  # 5cm
        if dist_to_goal < slowdown_radius:
            # Allow dropping to very low speeds
            brake_factor = max(0.1, dist_to_goal / slowdown_radius) 
        else:
            brake_factor = 1.0

        scaled_action = action * self.max_vel * brake_factor
        pos, vel = self.sim.run(scaled_action, num_steps=1) # 1 step = high freq
        
        new_dist = np.linalg.norm(pos - self.goal_pos)
        dist_mm = new_dist * 1000.0
        
        # 💰 Logarithmic Reward (Infinite gradient at 0)
        reward = -np.log(new_dist + 1e-5)
        
        # Parking Brake Penalty: If close, velocity must be zero.
        if new_dist < 0.01: # Inside 1cm
            vel_penalty = np.linalg.norm(vel) * 50.0 
            reward -= vel_penalty

        reward -= 0.1 # Existence penalty
        
        terminated = False
        if new_dist < 0.001: # 1mm Success
            reward += 100.0
            terminated = True
            self.success_history.append(1)
        
        truncated = self.steps >= 500
        if truncated: self.success_history.append(0)

        info = {"dist_mm": dist_mm, "radius_mm": self.curriculum_radius * 1000}
        self.prev_dist = new_dist
        return self._get_obs(pos, vel), float(reward), terminated, truncated, info

# ==========================================
# 📊 LOGGER
# ==========================================
class PerformanceLogger(BaseCallback):
    def __init__(self, check_freq=1000):
        super().__init__()
        self.check_freq = check_freq
        self.dist_buffer = []

    def _on_step(self) -> bool:
        if self.locals['dones'][0]:
            info = self.locals['infos'][0]
            self.dist_buffer.append(info.get('dist_mm', 0.0))
            
        if self.n_calls % self.check_freq == 0 and self.dist_buffer:
            avg_dist = np.mean(self.dist_buffer[-20:])
            radius = self.locals['infos'][0].get('radius_mm', 0)
            print(f"STEP {self.n_calls} | Radius: {radius:.1f}mm | Avg Error: {avg_dist:.2f}mm")
            self.dist_buffer = [] 
        return True

# ==========================================
# 🏁 MAIN
# ==========================================
def main():
    # ✅ ENABLED: Connected to your specific project
    task = Task.init(
        project_name='Mentor Group - Myrthe/Group 1', 
        task_name='OT2_Precision_Mode',
        output_uri=True 
    )
    
    env = DummyVecEnv([lambda: OT2PrecisionEnv(render=False)])
    
    # Tuned for exploitation (low entropy)
    model = PPO(
        "MlpPolicy", 
        env, 
        verbose=1, 
        learning_rate=3e-4,     
        n_steps=4096,           
        batch_size=128,
        ent_coef=0.0,           # Force exploitation
        gae_lambda=0.95,
        clip_range=0.2,
        tensorboard_log="./ppo_logs/"
    )
    
    callback = PerformanceLogger(check_freq=2000)
    print("Starting Training (Precision Mode)...")
    
    model.learn(total_timesteps=10_000_000, callback=callback)
    model.save("final_model_precision")

if __name__ == "__main__":
    main()