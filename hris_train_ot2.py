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

# Ensure Tensorboard
try:
    import tensorboard
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "tensorboard"])

# ==========================================
# 🚀 SIMULATION (Unchanged)
# ==========================================
class Simulation:
    def __init__(self, num_agents, render=False):
        self.physicsClient = p.connect(p.DIRECT)
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
            p.resetJointState(rId, joints['x'], 0.0)
            p.resetJointState(rId, joints['y'], 0.0)
            p.resetJointState(rId, joints['z'], 0.0)
        return self.get_states()

    def run(self, actions, num_steps=5):
        for _ in range(num_steps):
            for i, rId in enumerate(self.robotIds):
                joints = self.robot_joint_info[i]
                vx, vy, vz = actions[i]
                # Stable forces
                p.setJointMotorControl2(rId, joints['x'], p.VELOCITY_CONTROL, targetVelocity=vx, force=200)
                p.setJointMotorControl2(rId, joints['y'], p.VELOCITY_CONTROL, targetVelocity=vy, force=200)
                p.setJointMotorControl2(rId, joints['z'], p.VELOCITY_CONTROL, targetVelocity=vz, force=300)
            p.stepSimulation()
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
# 🧠 ENVIRONMENT (The Fix)
# ==========================================
class OT2Env(gym.Env):
    def __init__(self, render=False):
        super().__init__()
        self.sim = Simulation(num_agents=1, render=render)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
        
        # Added extra observation dimensions for "previous error" to help it see improvement
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(10,), dtype=np.float32)
        
        self.workspace_low = np.array([-0.1871, -0.1706, 0.1195], dtype=np.float32)
        self.workspace_high = np.array([0.2532, 0.2197, 0.2897], dtype=np.float32)
        self.workspace_span = self.workspace_high - self.workspace_low
        
        self.goal_pos = np.zeros(3)
        self.steps = 0
        self.prev_dist = 0.0
        self.last_action = np.zeros(3, dtype=np.float32)
        self.max_vel = 0.1

        # 🎓 AUTOMATIC CURRICULUM SETUP
        # Start SMALL. Master the "last inch" first.
        self.curriculum_radius = 0.05  # Start at 5cm (50mm)
        self.max_radius = 0.30         # Max 30cm (300mm)
        self.success_history = deque(maxlen=100) # Track last 100 episodes
        self.success_threshold_mm = 1.0 # The "Circle" target (1mm)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.steps = 0
        self.last_action = np.zeros(3, dtype=np.float32)
        
        # 1. Update Curriculum based on success rate
        if len(self.success_history) > 20:
            success_rate = np.mean(self.success_history)
            if success_rate > 0.85 and self.curriculum_radius < self.max_radius:
                self.curriculum_radius *= 1.05 # Grow circle by 5%
                self.curriculum_radius = min(self.curriculum_radius, self.max_radius)
            # Optional: Shrink if failing too much (prevents getting stuck)
            elif success_rate < 0.50 and self.curriculum_radius > 0.05:
                self.curriculum_radius *= 0.95

        states = self.sim.reset()
        curr_pos = np.array(states[0][0], dtype=np.float32)
        
        # 2. Spawn Goal within Current Circle of Mastery
        random_dir = self.np_random.uniform(-1, 1, size=3)
        random_dir /= np.linalg.norm(random_dir) + 1e-6
        # Dist is random between 1cm and current curriculum radius
        random_dist = self.np_random.uniform(0.01, self.curriculum_radius)
        
        self.goal_pos = curr_pos + (random_dir * random_dist)
        self.goal_pos = np.clip(self.goal_pos, self.workspace_low, self.workspace_high).astype(np.float32)
        
        self.prev_dist = np.linalg.norm(curr_pos - self.goal_pos)
        return self._get_obs(curr_pos, np.zeros(3)), {"goal": self.goal_pos}

    def _get_obs(self, pos, vel):
        rel_pos = self.goal_pos - pos
        norm_rel_pos = rel_pos / self.workspace_span 
        norm_vel = vel / self.max_vel 
        return np.concatenate([norm_rel_pos, norm_vel, self.last_action, [self.prev_dist]]).astype(np.float32)

    def step(self, action):
        self.steps += 1
        self.last_action = np.clip(action, -1.0, 1.0).astype(np.float32)
        
        # ⚙️ GEARBOX: Stronger braking near target
        # If dist < 5cm, slow down drastically to prevent "vibration"
        dist_factor = np.clip(self.prev_dist / 0.1, 0.1, 1.0) 
        vel_cmd = self.last_action * self.max_vel * dist_factor

        states = self.sim.run([vel_cmd], num_steps=5)
        new_pos = np.array(states[0][0], dtype=np.float32)
        new_vel = np.array(states[0][1], dtype=np.float32)
        
        # Boundary clip
        new_pos = np.clip(new_pos, self.workspace_low - 0.01, self.workspace_high + 0.01)
        curr_dist = np.linalg.norm(new_pos - self.goal_pos)

        # =======================
        # 🧠 SMOOTH GRADIENT REWARD
        # =======================
        # We removed the "steps" (magnets). Now it's a smooth hill.
        
        # 1. Progress Reward (Logarithmic is great for precision)
        # We want improvement at 10mm to matter as much as improvement at 100mm
        old_dist_mm = self.prev_dist * 1000
        new_dist_mm = curr_dist * 1000
        progress = (old_dist_mm - new_dist_mm) 
        
        # 2. Precision Bonus (Continuous, not stepped)
        # The closer you are, the higher the base value.
        # 1mm error = 1.0 reward per step, 10mm error = 0.1 reward
        precision_reward = 1.0 / (new_dist_mm + 1.0) 

        # 3. Penalties
        action_cost = -0.01 * np.linalg.norm(self.last_action)
        time_cost = -0.05
        
        reward = progress + precision_reward + action_cost + time_cost

        # Success Logic
        terminated = bool(curr_dist < (self.success_threshold_mm / 1000.0))
        truncated = self.steps >= 1000
        
        if terminated:
            reward += 50.0 # Big completion bonus
            self.success_history.append(1)
        elif truncated:
            self.success_history.append(0)

        self.prev_dist = curr_dist

        return self._get_obs(new_pos, new_vel), float(reward), terminated, truncated, {
            "distance": curr_dist, 
            "radius": self.curriculum_radius
        }

# ==========================================
# 🧠 TRAINING CALLBACKS
# ==========================================
class CurriculumLogger(BaseCallback):
    def __init__(self, check_freq=5000):
        super().__init__()
        self.check_freq = check_freq
        self.dist_buffer = []

    def _on_step(self) -> bool:
        if self.locals['dones'][0]:
            info = self.locals['infos'][0]
            dist_mm = info.get('distance', 1.0) * 1000
            radius_mm = info.get('radius', 0.0) * 1000
            
            self.dist_buffer.append(dist_mm)
            if len(self.dist_buffer) > 50: self.dist_buffer.pop(0)
            
            self.logger.record("curriculum/radius_mm", radius_mm)
            self.logger.record("curriculum/final_dist_mm", dist_mm)

        if self.n_calls % self.check_freq == 0 and self.dist_buffer:
            avg_dist = np.mean(self.dist_buffer)
            radius = self.locals['infos'][0].get('radius', 0) * 1000
            print(f"STEP {self.n_calls} | Avg Final Dist: {avg_dist:.2f}mm | Curr Radius: {radius:.1f}mm")
            gc.collect() 
        return True

def main():
    th.set_num_threads(4) 

    # Clean previous run
    if os.path.exists("./ppo_ot2_tensorboard/"):
        import shutil
        shutil.rmtree("./ppo_ot2_tensorboard/", ignore_errors=True)

    task = Task.init(
        project_name='Mentor Group - Myrthe/Group 1', 
        task_name='hris_Precision_Refinement_V3',
        output_uri=True 
    )
    task.execute_remotely(queue_name='default', exit_process=True)
    
    env = DummyVecEnv([lambda: OT2Env(render=False)])
    
    checkpoint_callback = CheckpointCallback(
        save_freq=100000, 
        save_path='./checkpoints/',
        name_prefix='ot2_precision_v3'
    )

    model = PPO(
        "MlpPolicy",
        env,
        device="cpu",       
        learning_rate=3e-4, # Reset to standard LR, let curriculum handle the difficulty
        n_steps=2048,
        batch_size=64,      
        n_epochs=10,        
        gamma=0.99,
        ent_coef=0.01,      # Increased slightly to prevent "stuck at 20mm" policy
        verbose=1,
        tensorboard_log="./ppo_ot2_tensorboard/"
    )
    
    callback_list = CallbackList([CurriculumLogger(check_freq=5000), checkpoint_callback])
    
    try:
        print("Starting Training (Expanding Circle Mode)...")
        # 5M steps is plenty for curriculum
        model.learn(total_timesteps=10_000_000, callback=callback_list) 
        model.save("final_model_precision_v3")
        task.upload_artifact("final_model", "final_model_precision_v3.zip")
    except Exception as e:
        print(f"Training interrupted: {e}")
        model.save("emergency_recovery_model")

if __name__ == "__main__":
    main()