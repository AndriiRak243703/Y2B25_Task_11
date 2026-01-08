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
# 🚀 SIMULATION (HARDCODED HEADLESS MODE)
# ==========================================
class Simulation:
    def __init__(self, num_agents, render=False):
        # 🔥 FIX: IGNORE the 'render' variable. Force DIRECT mode.
        # This guarantees 1000+ FPS by disabling all graphics context.
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
                p.setJointMotorControl2(rId, joints['x'], p.VELOCITY_CONTROL, targetVelocity=vx, force=500)
                p.setJointMotorControl2(rId, joints['y'], p.VELOCITY_CONTROL, targetVelocity=vy, force=500)
                p.setJointMotorControl2(rId, joints['z'], p.VELOCITY_CONTROL, targetVelocity=vz, force=800)
            p.stepSimulation()
        return self.get_states()

    def get_states(self):
        states = []
        for idx, rId in enumerate(self.robotIds):
            joints = self.robot_joint_info[idx]
            # Fetch all 3 joints in one go (Fastest method)
            js = p.getJointStates(rId, [joints['x'], joints['y'], joints['z']])
            
            pos = [
                js[0][0] + self.pipette_offset[0],
                js[1][0] + self.pipette_offset[1],
                js[2][0] + self.pipette_offset[2]
            ]
            vel = [js[0][1], js[1][1], js[2][1]]
            states.append((pos, vel))
        return states

    def close(self):
        p.disconnect()


class OT2Env(gym.Env):
    def __init__(self, render=False):
        super().__init__()
        # Note: Render arg is passed but ignored by Simulation class now
        self.sim = Simulation(num_agents=1, render=render)
        
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-2.0, high=2.0, shape=(13,), dtype=np.float32)
        
        self.workspace_low = np.array([-0.1871, -0.1706, 0.1195], dtype=np.float32)
        self.workspace_high = np.array([0.2532, 0.2197, 0.2897], dtype=np.float32)
        
        self.goal_pos = np.zeros(3)
        self.steps = 0
        self.prev_dist = 0.0
        self.last_action = np.zeros(3, dtype=np.float32)
        self.max_vel = 0.1 

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.steps = 0
        self.last_action = np.zeros(3, dtype=np.float32)
        self.goal_pos = self.np_random.uniform(self.workspace_low, self.workspace_high).astype(np.float32)
        
        states = self.sim.reset()
        curr_pos = np.array(states[0][0], dtype=np.float32)
        curr_pos = np.clip(curr_pos, self.workspace_low, self.workspace_high)
        
        self.prev_dist = np.linalg.norm(curr_pos - self.goal_pos)
        return self._get_obs(curr_pos, np.zeros(3)), {"goal": self.goal_pos}

    def _get_obs(self, pos, vel):
        norm_pos = 2.0 * (pos - self.workspace_low) / (self.workspace_high - self.workspace_low) - 1.0
        norm_goal = 2.0 * (self.goal_pos - self.workspace_low) / (self.workspace_high - self.workspace_low) - 1.0
        norm_vel = vel / self.max_vel 
        return np.concatenate([norm_pos, norm_goal, norm_vel, self.last_action, [self.prev_dist]]).astype(np.float32)

    def step(self, action):
        self.steps += 1
        self.last_action = np.clip(action, -1.0, 1.0).astype(np.float32)
        
        # Lock-In Physics: Slow down when close
        max_vel = min(0.1, 0.3 * self.prev_dist)
        vel_cmd = self.last_action * max_vel

        states = self.sim.run([vel_cmd], num_steps=5)
        new_pos = np.array(states[0][0], dtype=np.float32)
        new_vel = np.array(states[0][1], dtype=np.float32)
        
        new_pos = np.clip(new_pos, self.workspace_low - 0.01, self.workspace_high + 0.01)
        curr_dist = np.linalg.norm(new_pos - self.goal_pos)

        # Rewards
        progress = (self.prev_dist - curr_dist) * 500.0
        proximity = min(1.0 / (curr_dist + 0.01), 50.0) # Clamped
        
        reward = progress + proximity - 0.01 - (0.1 * np.linalg.norm(self.last_action))

        terminated = bool(curr_dist < 0.001)
        if terminated:
            reward += 100.0

        truncated = self.steps >= 1000
        self.prev_dist = curr_dist

        return self._get_obs(new_pos, new_vel), float(reward), terminated, truncated, {"distance": curr_dist}

# ==========================================
# 🧠 TRAINING (FIXED FPS & VISIBLE MM)
# ==========================================
class PrecisionLRScheduler(BaseCallback):
    def __init__(self, check_freq=5000):
        super().__init__()
        self.check_freq = check_freq
        self.precision_buffer = []
        self.current_lr = 2.5e-4

    def _on_step(self) -> bool:
        if self.locals['dones'][0]:
            dist_mm = self.locals['infos'][0].get('distance', 1.0) * 1000
            self.precision_buffer.append(dist_mm)
            if len(self.precision_buffer) > 50:
                self.precision_buffer.pop(0)

        # 🔥 FPS FIX: GC and Logging ONLY run here (every ~10 seconds)
        if self.n_calls % self.check_freq == 0 and self.precision_buffer:
            avg_dist = np.mean(self.precision_buffer)
            self.logger.record("trajectory/avg_distance_mm", avg_dist)
            
            # Print status to logs
            print(f"STATUS: Step {self.n_calls} | Avg Dist: {avg_dist:.2f}mm | LR: {self.current_lr:.1e}")
            
            if avg_dist < 1.0: target_lr = 5e-6
            elif avg_dist < 2.0: target_lr = 1e-5
            elif avg_dist < 5.0: target_lr = 2e-5
            elif avg_dist < 15.0: target_lr = 5e-5
            elif avg_dist < 30.0: target_lr = 7.5e-5
            elif avg_dist < 45.0: target_lr = 1e-4
            elif avg_dist < 60.0: target_lr = 1.75e-4
            else: target_lr = 2.5e-4

            if target_lr < self.current_lr:
                self.current_lr = target_lr
                for param_group in self.model.policy.optimizer.param_groups:
                    param_group['lr'] = self.current_lr
                print(f"MILESTONE REACHED: Dropping LR to {self.current_lr:.1e}")
            
            self.logger.record("train/learning_rate_dynamic", self.current_lr)
            gc.collect() 
            
        return True

def main():
    task = Task.init(
        project_name='Mentor Group - Myrthe/Group 1', 
        task_name='hris_Precision_Final_Fixed_V4',
        output_uri=True 
    )
    task.execute_remotely(queue_name='default', exit_process=True)
    
    # 2. CREATE ENVIRONMENT (Render arg is now irrelevant, it is forced OFF)
    env = DummyVecEnv([lambda: OT2Env(render=False)])
    
    checkpoint_callback = CheckpointCallback(
        save_freq=200000, 
        save_path='./checkpoints/',
        name_prefix='ot2_curriculum_v4'
    )

    model = PPO(
        "MlpPolicy",
        env,
        device="cpu",
        learning_rate=2.5e-4,
        n_steps=2048,
        batch_size=64,
        gamma=0.995,
        ent_coef=0.001,
        verbose=1,
        tensorboard_log="./ppo_ot2_tensorboard/"
    )
    
    callback_list = CallbackList([PrecisionLRScheduler(check_freq=5000), checkpoint_callback])
    
    try:
        print("Starting Training (Hardcoded Headless Mode)...")
        model.learn(total_timesteps=10_000_000, callback=callback_list)
        model.save("final_model_v4")
        task.upload_artifact("final_model_v4", "final_model_v4.zip")
    except Exception as e:
        print(f"Training interrupted: {e}")
        model.save("emergency_recovery_model")
        task.upload_artifact("crash_model", "emergency_recovery_model.zip")

if __name__ == "__main__":
    main()