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
BRANCH_NAME = "hris"
DEBUG_MODE = True  # ✅ Set to True for Local Visuals, False for ClearML Cloud

# ==============================
# 📦 CLEARML SETUP
# ==============================
task = Task.init(
    project_name='Mentor Group - Myrthe/Group 1',
    task_name=f'OT2_PPO_DenseReward_{PERSON_NAME}',
    output_uri=True,
    reuse_last_task_id=False
)

task.set_base_docker('deanis/2023y2b-rl:latest')
task.set_repo(repo='https://github.com/AndriiRak243703/Y2B25_Task_11.git', branch=BRANCH_NAME)
task.set_packages(['stable-baselines3', 'gymnasium', 'pybullet', 'numpy', 'clearml', 'tensorboard'])

# Hyperparameters
parser = argparse.ArgumentParser()
parser.add_argument("--learning_rate", type=float, default=3e-4)
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--n_steps", type=int, default=2048)
parser.add_argument("--total_timesteps", type=int, default=1_000_000)
parser.add_argument("--gamma", type=float, default=0.99)
args, unknown = parser.parse_known_args()

# Execute remotely ONLY if not debugging
if not DEBUG_MODE:
    task.execute_remotely(queue_name='default')

# ==============================
# 🧪 ENVIRONMENT WRAPPER
# ==============================
def make_wrapped_env(render=False):
    """
    Creates the OT2 environment.
    :param render: If True, opens the PyBullet GUI.
    """
    env = OT2Env(render=render)
    env = TimeLimit(env, max_episode_steps=1000)
    # Monitor logs success rate and distance to ClearML/Tensorboard
    env = Monitor(env, info_keywords=("is_success", "dist_mm"))
    return env

# ==============================
# 🏁 MAIN TRAINING
# ==============================
def main():
    mode_str = "LOCALLY (Debug)" if DEBUG_MODE else "REMOTELY (Cloud)"
    print(f"🚀 Starting PPO training {mode_str} for {PERSON_NAME}")
    
    # 1. Training Environment
    # If Debugging: 1 env, Render=True (Visuals)
    # If Cloud: 4 envs, Render=False (Fast)
    num_envs = 1 if DEBUG_MODE else 4
    
    # We use a lambda to pass the specific 'render' flag to the vector env
    env = make_vec_env(lambda: make_wrapped_env(render=DEBUG_MODE), n_envs=num_envs, seed=42)
    
    # 2. Evaluation Environment
    # ⚠️ CRITICAL FIX: Always set render=False here.
    # PyBullet crashes if you try to open a second GUI window.
    eval_env = make_wrapped_env(render=False)
    
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=f"./models/{PERSON_NAME}/",
        log_path=f"./logs/{PERSON_NAME}/",
        eval_freq=max(5000 // num_envs, 1),
        n_eval_episodes=10,
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

    print(f"Initialized PPO. Training for {args.total_timesteps} timesteps...")
    
    try:
        model.learn(
            total_timesteps=args.total_timesteps,
            callback=eval_callback,
            progress_bar=True,
        )
    except KeyboardInterrupt:
        print("Training interrupted by user. Saving current state...")

    # Save final model
    model_name = f"ppo_ot2_{PERSON_NAME}.zip"
    model.save(model_name)
    print(f"✅ Saved model: {model_name}")
    task.upload_artifact("final_model", artifact_object=model_name)

if __name__ == "__main__":
    main()