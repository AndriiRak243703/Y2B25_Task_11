import os
import argparse
import numpy as np
import torch
from clearml import Task

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from gymnasium.wrappers import TimeLimit
from her_wrapper import OT2Env

# ==============================
# 🔧 CONFIGURATION
# ==============================
PERSON_NAME = "241236"
DEBUG_MODE = True  # SET TO FALSE TO RUN ON THE SERVER

# ==============================
# 📦 CLEARML SETUP
# ==============================
task = Task.init(
    project_name='Mentor Group - Myrthe/Group 1',
    task_name=f'OT2_PPO_DenseReward_{PERSON_NAME}',
    output_uri=True
)

if not DEBUG_MODE:
    task.execute_remotely(queue_name='default')
    task.set_base_docker('deanis/2023y2b-rl:latest')
    task.set_packages(['stable-baselines3', 'gymnasium', 'pybullet', 'numpy'])

# Hyperparameters
parser = argparse.ArgumentParser()
parser.add_argument("--learning_rate", type=float, default=3e-4)
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--n_steps", type=int, default=2048)
parser.add_argument("--total_timesteps", type=int, default=1_000_000)
args, _ = parser.parse_known_args()

# ==============================
# 🧪 HELPER FUNCTIONS
# ==============================
def make_wrapped_env():
    # Only render if we are debugging locally
    env = OT2Env(render=DEBUG_MODE)
    env = Monitor(env, info_keywords=("is_success", "dist_mm"))
    return env

def main():
    print(f"🚀 Training starting. Debug Mode: {DEBUG_MODE}")
    
    # Environments
    num_envs = 1 if DEBUG_MODE else 4
    env = make_vec_env(make_wrapped_env, n_envs=num_envs)
    
    # Eval Callback
    eval_env = make_wrapped_env()
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=f"./models/{PERSON_NAME}/",
        log_path=f"./logs/{PERSON_NAME}/",
        eval_freq=max(10000 // num_envs, 1),
        n_eval_episodes=5,
        deterministic=True,
        verbose=1,
    )

    # Model
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=args.learning_rate,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        verbose=1,
        tensorboard_log="./ppo_tensorboard/"
    )

    # Learn
    model.learn(
        total_timesteps=args.total_timesteps,
        callback=eval_callback,
        progress_bar=True
    )

    # Save
    model.save(f"ppo_ot2_{PERSON_NAME}")
    task.upload_artifact("final_model", artifact_object=f"ppo_ot2_{PERSON_NAME}.zip")

if __name__ == "__main__":
    main()