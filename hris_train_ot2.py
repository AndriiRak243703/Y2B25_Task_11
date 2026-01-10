import numpy as np
import gymnasium as gym
from gymnasium import spaces
import pybullet as p
import pybullet_data
import time
import sys
import subprocess
from gymnasium.wrappers import TimeLimit

# Install tensorboard if missing
try:
    import tensorboard
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "tensorboard"])

# ClearML (optional)
try:
    from clearml import Task
    if Task.current_task() is None:
        task = Task.init(
            project_name='Mentor Group - Myrthe/Group 1',
            task_name='OT2_SAC_HER_1mm',
            output_uri=True
        )
    else:
        task = Task.current_task()
except ImportError:
    print("ClearML not installed. Continuing without logging.")
    task = None

from stable_baselines3 import SAC
from stable_baselines3.her import HerReplayBuffer
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback


# ==========================================
# 📊 CUSTOM LOGGER (To see mm error live)
# ==========================================
class HumanReadableLogCallback(BaseCallback):
    def _on_step(self) -> bool:
        # Check if we have any finished episodes in the buffer
        if len(self.model.ep_info_buffer) > 0:
            # We look for 'dist_mm' which we ensured is saved by monitor_kwargs
            distances = [ep_info["dist_mm"] for ep_info in self.model.ep_info_buffer if "dist_mm" in ep_info]
            if len(distances) > 0:
                self.logger.record("rollout/dist_mm_mean", np.mean(distances))
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
            pass  # Already connected
            
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, 0)
        p.loadURDF("plane.urdf")
        self.pipette_offset = [0.073, 0.0895, 0.0895]
        self.robotId = p.loadURDF("ot_2_simulation_v6.urdf", [0, 0, 0], useFixedBase=True)
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
                time.sleep(1. / 240.)
        return self.get_state()

    def get_state(self):
        js = p.getJointStates(self.robotId, self.joints)
        pos = [js[0][0] + self.pipette_offset[0], js[1][0] + self.pipette_offset[1], js[2][0] + self.pipette_offset[2]]
        vel = [js[0][1], js[1][1], js[2][1]]
        return np.array(pos), np.array(vel)


# ==========================================
# 🧠 GOAL-CONDITIONED ENVIRONMENT
# ==========================================
class OT2PrecisionEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, render=False):
        super().__init__()
        self.render_mode = "human" if render else None
        self.sim = Simulation(render=render)
        self.max_vel = 0.1
        self.distance_threshold = 0.001  # 1mm

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)

        obs_dim = 3
        self.observation_space = spaces.Dict({
            "observation": spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32),
            "achieved_goal": spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32),
            "desired_goal": spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32),
        })

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            np.random.seed(seed)
        pos, _ = self.sim.reset()

        # Random goal 1cm to 20cm away
        rand_dist = self.np_random.uniform(0.01, 0.20)
        direction = self.np_random.uniform(-1, 1, size=3)
        norm = np.linalg.norm(direction)
        if norm > 0:
            direction /= norm
        self.goal = pos + direction * rand_dist

        obs = {
            "observation": pos.astype(np.float32),
            "achieved_goal": pos.astype(np.float32),
            "desired_goal": self.goal.astype(np.float32),
        }
        return obs, {}

    def step(self, action):
        scaled_action = action * self.max_vel
        pos, vel = self.sim.run(scaled_action, num_steps=1)

        achieved_goal = pos
        desired_goal = self.goal
        distance = np.linalg.norm(achieved_goal - desired_goal)

        # Sparse reward: 0 if success, -1 otherwise
        reward = 0.0 if distance < self.distance_threshold else -1.0
        terminated = distance < self.distance_threshold
        truncated = False  # Handled by wrapper

        obs = {
            "observation": pos.astype(np.float32),
            "achieved_goal": achieved_goal.astype(np.float32),
            "desired_goal": desired_goal.astype(np.float32),
        }
        info = {
            "is_success": float(terminated),
            "dist_mm": distance * 1000.0,
        }
        return obs, reward, terminated, truncated, info

    def compute_reward(self, achieved_goal, desired_goal, info):
        distance = np.linalg.norm(achieved_goal - desired_goal, axis=-1)
        return -(distance >= self.distance_threshold).astype(np.float32)


# ==========================================
# 🏁 MAIN TRAINING LOOP
# ==========================================
def main():
    print("🚀 Starting SAC + HER training for 1mm precision...")

    # 1. Create Env with monitor_kwargs to track 'dist_mm'
    env = make_vec_env(
        OT2PrecisionEnv, 
        n_envs=4, 
        seed=42, 
        env_kwargs={'render': False},
        wrapper_class=TimeLimit,
        wrapper_kwargs={'max_episode_steps': 1000},
        monitor_kwargs={'info_keywords': ('dist_mm',)} 
    )

    # 2. Setup Eval Env
    eval_env = TimeLimit(OT2PrecisionEnv(render=False), max_episode_steps=1000)
    
    # 3. Callbacks
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
    callbacks = [eval_callback, logger_callback]

    # 4. Initialize Model
    model = SAC(
        "MultiInputPolicy",
        env,
        replay_buffer_class=HerReplayBuffer,
        replay_buffer_kwargs=dict(
            n_sampled_goal=4,
            goal_selection_strategy="future",
        ),
        verbose=1,
        buffer_size=500_000,
        learning_rate=1e-3,
        gamma=0.99,
        tau=0.005,
        train_freq=1,
        gradient_steps=1,
        learning_starts=10000,  # CRITICAL: Prevents crash with 4 envs
        ent_coef="auto",
        tensorboard_log="./sac_her_logs/"
    )

    print("Initialized SAC+HER. Training for 5M timesteps...")
    model.learn(
        total_timesteps=5_000_000,
        callback=callbacks,
        log_interval=10,
        progress_bar=True
    )

    model.save("ot2_sac_her_1mm_final")
    print("✅ Training complete.")

    # 5. Final Test
    print("\n🧪 Final evaluation (100 episodes):")
    success_count = 0
    errors = []
    
    for _ in range(100):
        obs, _ = eval_env.reset()
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, info = eval_env.step(action)
            done = terminated or truncated
        
        if "dist_mm" in info:
            errors.append(info["dist_mm"])
        if info.get("is_success", 0.0) > 0.0:
            success_count += 1

    success_rate = success_count / 100 * 100
    median_error = np.median(errors) if errors else 0.0
    print(f"Success @ 1mm: {success_rate:.1f}%")
    print(f"Median Error: {median_error:.2f} mm")

    if task:
        task.get_logger().report_scalar("Final", "Success_Rate_1mm", success_rate, iteration=0)
        task.get_logger().report_scalar("Final", "Median_Error_mm", median_error, iteration=0)


if __name__ == "__main__":
    main()