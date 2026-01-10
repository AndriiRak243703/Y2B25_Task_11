import os
import argparse
import numpy as np
import torch

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from gymnasium.wrappers import TimeLimit
from her_wrapper import OT2Env

# ==============================
# 🔧 CONFIGURATION (EDIT THESE)
# ==============================
PERSON_NAME = "241236"          # e.g., "myrthe"
BRANCH_NAME = "hris"        # e.g., "myrthe_branch"
ENTRYPOINT = '241236_train_ot2.py'

# Import your custom environment

# ==============================
# 📦 CLEARML SETUP
# ==============================
from clearml import Task

task = Task.init(
    project_name='Mentor Group - Myrthe/Group 1',
    task_name=f'OT2_PPO_DenseReward_{PERSON_NAME}',
    output_uri=True
)
task.set_base_docker('deanis/2023y2b-rl:latest')
task.set_repo(repo='https://github.com/AndriiRak243703/Y2B25_Task_11.git', branch=BRANCH_NAME)
task.set_packages(['stable-baselines3', 'gymnasium', 'pybullet', 'numpy', 'clearml', 'tensorboard'])

# Parse args BEFORE execute_remotely
parser = argparse.ArgumentParser()
parser.add_argument("--learning_rate", type=float, default=3e-4)
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--n_steps", type=int, default=2048)
parser.add_argument("--total_timesteps", type=int, default=1_000_000)
parser.add_argument("--gamma", type=float, default=0.99)
args = parser.parse_args()

task.execute_remotely(queue_name='default')

# ==============================
# 🧪 ENVIRONMENT WRAPPER
# ==============================
def make_wrapped_env(render=False):
    env = OT2Env(render=render)
    env = TimeLimit(env, max_episode_steps=1000)
    env = Monitor(env, info_keywords=("is_success", "dist_mm"))
    return env

# ==============================
# 🏁 MAIN TRAINING
# ==============================
def main():
    print(f"🚀 Starting PPO training for {PERSON_NAME} on branch {BRANCH_NAME}")
    
    # Vectorized training envs
    env = make_vec_env(make_wrapped_env, n_envs=4, seed=42)
    
    # Evaluation env
    eval_env = make_wrapped_env(render=False)
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=f"./models/{PERSON_NAME}/",
        log_path=f"./logs/{PERSON_NAME}/",
        eval_freq=max(5000 // 4, 1),  # adjust for vec envs
        n_eval_episodes=20,
        deterministic=True,
        render=False,
        verbose=1,
    )

    # PPO Model
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=args.learning_rate,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        gamma=args.gamma,
        gae_lambda=0.95,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        tensorboard_log="./ppo_tensorboard/",
        verbose=1,
    )

    print(f"Intialized PPO. Training for {args.total_timesteps} timesteps...")
    model.learn(
        total_timesteps=args.total_timesteps,
        callback=eval_callback,
        progress_bar=False,
    )

    # Save final model
    timestamp = task.get_task_start_time().strftime("%y%m%d.%H%M")
    model_name = f"{timestamp}_{PERSON_NAME}_lr{args.learning_rate:.0e}_b{args.batch_size}_s{args.n_steps}.zip"
    model.save(model_name)
    print(f"✅ Saved model: {model_name}")
    task.upload_artifact("final_model", artifact_object=model_name)

if __name__ == "__main__":
    main()