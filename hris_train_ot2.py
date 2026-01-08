import numpy as np
import gymnasium as gym
from gymnasium import spaces
import pybullet as p
import pybullet_data
import math
import sys
import subprocess
import torch as th
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
# 🚀 SIMULATION
# ==========================================
class Simulation:
    def __init__(self, num_agents, render=False):
        self.render = render
        self.mode = p.GUI if render else p.DIRECT
        self.physicsClient = p.connect(self.mode)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, 0) # ⚠️ NO GRAVITY (Simplifies movement for OT2)
        self.planeId = p.loadURDF("plane.urdf")
        self.pipette_offset = [0.073, 0.0895, 0.0895]
        self.robotIds = []
        self.robot_joint_info = []
        self.create_robots(num_agents)

    def create_robots(self, num_agents):
        self.robotIds.append(p.loadURDF("ot_2_simulation_v6.urdf", [0,0,0], useFixedBase=True))
        # Map joints
        self.robot_joint_info.append({'x': 0, 'y': 1, 'z': 2}) # Assuming 0,1,2 are the sliders based on standard URDFs

    def reset(self):
        for rId in self.robotIds:
            # Reset to slightly random center to prevent overfitting
            p.resetJointState(rId, 0, np.random.uniform(-0.01, 0.01))
            p.resetJointState(rId, 1, np.random.uniform(-0.01, 0.01))
            p.resetJointState(rId, 2, np.random.uniform(-0.01, 0.01))
        return self.get_states()

    def run(self, actions, num_steps=5):
        for _ in range(num_steps):
            vx, vy, vz = actions[0]
            # Hard Velocity Control - Direct and Strong
            p.setJointMotorControl2(self.robotIds[0], 0, p.VELOCITY_CONTROL, targetVelocity=vx, force=1000)
            p.setJointMotorControl2(self.robotIds[0], 1, p.VELOCITY_CONTROL, targetVelocity=vy, force=1000)
            p.setJointMotorControl2(self.robotIds[0], 2, p.VELOCITY_CONTROL, targetVelocity=vz, force=1000)
            p.stepSimulation()
        return self.get_states()

    def get_states(self):
        js = p.getJointStates(self.robotIds[0], [0, 1, 2])
        # Simple Position (Pipette Tip)
        pos = [js[0][0] + self.pipette_offset[0], js[1][0] + self.pipette_offset[1], js[2][0] + self.pipette_offset[2]]
        vel = [js[0][1], js[1][1], js[2][1]]
        return [(pos, vel)]

# ==========================================
# 🧠 ENVIRONMENT
# ==========================================
class OT2ExpandingEnv(gym.Env):
    def __init__(self, render=False):
        super().__init__()
        self.sim = Simulation(num_agents=1, render=render)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32)
        
        self.goal_pos = np.zeros(3)
        self.steps = 0
        self.max_vel = 0.5 # Give it speed
        
        # 🎓 CURRICULUM
        self.curriculum_radius = 0.05 # Start at 5cm (Easier to see progress than 2cm)
        self.success_history = deque(maxlen=50)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.steps = 0

        # Auto-Expand
        if len(self.success_history) >= 20 and np.mean(self.success_history) > 0.8:
            self.curriculum_radius = min(self.curriculum_radius * 1.2, 0.30)
            self.success_history.clear()
            print(f"🚀 EXPANDING! Radius: {self.curriculum_radius*1000:.1f} mm")

        states = self.sim.reset()
        curr_pos = np.array(states[0][0], dtype=np.float32)
        
        # Spawn Goal
        random_dir = self.np_random.uniform(-1, 1, size=3)
        random_dir /= np.linalg.norm(random_dir) + 1e-6
        self.goal_pos = curr_pos + (random_dir * self.curriculum_radius)
        
        return self._get_obs(curr_pos, np.zeros(3)), {}

    def _get_obs(self, pos, vel):
        rel_pos = self.goal_pos - pos
        # BIG SCALING: Make the inputs visible to the net
        return np.concatenate([rel_pos * 100.0, vel * 10.0]).astype(np.float32)

    def step(self, action):
        self.steps += 1
        action = np.clip(action, -1.0, 1.0).astype(np.float32)
        
        # Run Sim
        states = self.sim.run([action * self.max_vel])
        new_pos = np.array(states[0][0], dtype=np.float32)
        new_vel = np.array(states[0][1], dtype=np.float32)
        
        # Distances
        dist = np.linalg.norm(new_pos - self.goal_pos)
        
        # =======================
        # 🧠 HYBRID REWARD
        # =======================
        # 1. Negative Distance (The Funnel)
        reward = -dist * 10.0 
        
        # 2. Directional cosine (The "Homing Beacon")
        # "Are you pressing the stick in the direction of the goal?"
        vec_to_goal = self.goal_pos - new_pos
        vec_to_goal /= (np.linalg.norm(vec_to_goal) + 1e-6)
        
        # Alignment = Dot Product of Action and Direction
        # If action aligns with goal direction -> High Reward
        alignment = np.dot(action, vec_to_goal) 
        reward += alignment * 1.0 
        
        # 3. Success
        terminated = False
        if dist < 0.005: # 5mm threshold
            reward += 20.0
            terminated = True
            self.success_history.append(1)
        
        truncated = self.steps >= 200
        if truncated: self.success_history.append(0)

        return self._get_obs(new_pos, new_vel), float(reward), terminated, truncated, {"dist": dist, "radius": self.curriculum_radius}

# ==========================================
# 🏁 TRAINING
# ==========================================
def main():
    Task.init(project_name='Mentor Group', task_name='OT2_Directional_V1')
    env = DummyVecEnv([lambda: OT2ExpandingEnv(render=False)])
    
    # High learning rate + low entropy to force it to follow the gradient
    model = PPO("MlpPolicy", env, verbose=1, learning_rate=1e-3, ent_coef=0.0, tensorboard_log="./ppo_logs/")
    
    model.learn(total_timesteps=500_000)
    model.save("ot2_directional")

if __name__ == "__main__":
    main()