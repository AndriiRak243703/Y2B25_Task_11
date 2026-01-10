import numpy as np
import gymnasium as gym
from gymnasium import spaces
import pybullet as p
import pybullet_data
import time
import sys
import os
import argparse
import torch
from gymnasium.wrappers import TimeLimit
from stable_baselines3.common.monitor import Monitor
from stable_baselines3 import SAC
from stable_baselines3.her import HerReplayBuffer
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback
from clearml import Task

# ==========================================
# ⚡ PERFORMANCE TUNING
# ==========================================
torch.set_num_threads(2)

# ==========================================
# ⚙️ ARGS (Adjustable from ClearML UI)
# ==========================================
parser = argparse.ArgumentParser()
parser.add_argument("--learning_rate", type=float, default=1e-3)
parser.add_argument("--batch_size", type=int, default=256)
parser.add_argument("--total_timesteps", type=int, default=5_000_000)
parser.add_argument("--gamma", type=float, default=0.99)
parser.add_argument("--tau", type=float, default=0.005)
parser.add_argument("--train_freq", type=int, default=64)
parser.add_argument("--gradient_steps", type=int, default=32)
parser.add_argument("--learning_starts", type=int, default=10000)
args = parser.parse_args()

# ==========================================
# ☁️ CLEARML REMOTE SETUP
# ==========================================
task = Task.init(
    project_name='Mentor Group - Myrthe/Group 1',
    task_name='OT2_SAC_HER_Server_Run',
    output_uri=True
)

task.set_base_docker('deanis/2023y2b-rl:latest')

# ⚠️ CORRECT REPO AND BRANCH
task.set_repo(
    repo='https://github.com/AndriiRak243703/Y2B25_Task_11.git', 
    branch='hris'
)

task.set_packages(['stable-baselines3', 'gymnasium', 'tensorboard', 'pybullet', 'shimmy', 'clearml'])

# Send to remote queue
task.execute_remotely(queue_name='default')

print("🚀 Task sent to queue! Computing on server...")

# ==========================================
# 📊 AGGRESSIVE LOGGER
# ==========================================
class HumanReadableLogCallback(BaseCallback):
    def _on_step(self) -> bool:
        if len(self.model.ep_info_buffer) > 0:
            distances = [ep_info["dist_mm"] for ep_info in self.model.ep_info_buffer if "dist_mm" in ep_info]
            if len(distances) > 0:
                mean_dist = np.mean(distances)
                self.logger.record("rollout/dist_mm_mean", mean_dist)
        return True

# ==========================================
# 🚀 SIMULATION
# ==========================================
class Simulation:
    def __init__(self, render=False):
        self.render = render
        mode = p.GUI if render else p.DIRECT
        try:
            self.physicsClient = p.connect(mode)
        except p.error:
            pass
            
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, 0)
        
        # 🛡️ ROBUST PATH FINDING FOR SERVER
        # Finds URDFs relative to this script, wherever it is located
        script_dir = os.path.dirname(os.path.abspath(__file__))
        plane_path = os.path.join(script_dir, "plane.urdf")
        robot_path = os.path.join(script_dir, "ot_2_simulation_v6.urdf")

        # Try robust path first, fallback to simple filename if needed
        try:
            p.loadURDF(plane_path)
            self.robotId = p.loadURDF(robot_path, [0, 0, 0], useFixedBase=True)
        except:
            print("⚠️ Warning: Could not find URDFs at absolute path. Trying relative...")
            p.loadURDF("plane.urdf")
            self.robotId = p.loadURDF("ot_2_simulation_v6.urdf", [0, 0, 0], useFixedBase=True)

        self.pipette_offset = [0.073, 0.0895, 0.0895]
        self.joints = [0, 1, 2]

    def reset(self):
        start_pos = np.random.uniform(-0.15, 0.15, 3)
        start_pos[2] = np.random.uniform(0.05, 0.15) 
        
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
                time.sleep(1. / 240.)
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
    metadata = {"render_modes": ["human"]}

    def __init__(self, render=False):
        super().__init__()
        self.render_mode = None 
        self.sim = Simulation(render=False)
        
        self.max_vel = 0.2 
        self.distance_threshold = 0.01  # 1cm goal

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
        obs_dim = 3
        self.observation_space = spaces.Dict({
            "observation": spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32),
            "achieved_goal": spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32),
            "desired_goal": spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32),
        })

    def _get_obs(self, pos, goal):
        return {
            "observation": pos.astype(np.float32),
            "achieved_goal": pos.astype(np.float32),
            "desired_goal": goal.astype(np.float32),
        }

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            np.random.seed(seed)
        pos, _ = self.sim.reset()

        rand_dist = self.np_random.uniform(0.01, 0.20)
        direction = self.np_random.uniform(-1, 1, size=3)
        direction[2] = abs(direction[2]) 
        direction /= np.linalg.norm(direction)
        
        self.goal = pos + direction * rand_dist
        
        # 🛡️ SAFETY CLAMP: Fix for "Ghost Goals"
        self.goal[0] = np.clip(self.goal[0], -0.25, 0.25)
        self.goal[1] = np.clip(self.goal[1], -0.25, 0.25)
        self.goal[2] = np.clip(self.goal[2], 0.0, 0.35)

        return self._get_obs(pos, self.goal), {}

    def step(self, action):
        scaled_action = action * self.max_vel
        pos, vel = self.sim.run(scaled_action, num_steps=1)

        # 🚧 BOUNDARY CHECK
        out_of_bounds = False
        if pos[2] < 0.0: out_of_bounds = True
        if not (-0.3 < pos[0] < 0.3) or not (-0.3 < pos[1] < 0.3) or (pos[2] > 0.4): out_of_bounds = True

        achieved_goal = pos
        desired_goal = self.goal
        distance = np.linalg.norm(achieved_goal - desired_goal)

        if out_of_bounds:
            reward = -10.0 
            terminated = True 
            truncated = False
            info = {"is_success": 0.0, "dist_mm": distance * 1000.0}
            return self._get_obs(pos, desired_goal), reward, terminated, truncated, info

        if distance < self.distance_threshold:
            reward = 0.0
            terminated = True
            is_success = 1.0
        else:
            reward = -1.0
            terminated = False
            is_success = 0.0

        info = {"is_success": is_success, "dist_mm": distance * 1000.0}
        return self._get_obs(pos, desired_goal), reward, terminated, False, info

    def compute_reward(self, achieved_goal, desired_goal, info):
        distance = np.linalg.norm(achieved_goal - desired_goal, axis=-1)
        return -(distance >= self.distance_threshold).astype(np.float32)

# ==========================================
# 🛠️ HELPER
# ==========================================
def make_wrapped_env(render=False):
    env = OT2PrecisionEnv(render=render)
    env = TimeLimit(env, max_episode_steps=1000)
    env = Monitor(env, info_keywords=("dist_mm", "is_success")) 
    return env

# ==========================================
# 🏁 MAIN
# ==========================================
def main():
    print("🚀 Starting SAC + HER Training (Remote Mode)...")
    
    env = make_vec_env(make_wrapped_env, n_envs=4, seed=42, env_kwargs={'render': False})
    eval_env = make_wrapped_env(render=False)
    
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="./logs/",
        log_path="./logs/",
        eval_freq=25000,
        deterministic=True,
        render=False,
        n_eval_episodes=50,
        verbose=1,
    )
    logger_callback = HumanReadableLogCallback()
    
    model = SAC(
        "MultiInputPolicy",
        env,
        replay_buffer_class=HerReplayBuffer,
        replay_buffer_kwargs=dict(n_sampled_goal=4, goal_selection_strategy="future"),
        verbose=1,
        buffer_size=500_000,
        learning_rate=args.learning_rate,
        gamma=args.gamma,
        tau=args.tau,
        train_freq=args.train_freq,      
        gradient_steps=args.gradient_steps,  
        learning_starts=args.learning_starts, 
        ent_coef="auto",
        tensorboard_log="./sac_her_logs/"
    )

    print(f"Initialized. Training for {args.total_timesteps} steps...")
    model.learn(
        total_timesteps=args.total_timesteps,
        callback=[eval_callback, logger_callback],
        log_interval=10,
        progress_bar=False 
    )
    
    model_name = "ot2_sac_her_final.zip"
    model.save(model_name)
    print("✅ Training complete. Uploading artifact...")
    task.upload_artifact("model", artifact_object=model_name)
    print("✅ Artifact uploaded!")

if __name__ == "__main__":
    main()