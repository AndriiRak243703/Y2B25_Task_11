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
# ⚠️ SET TO FALSE FOR MAX SPEED ON SERVER
DEBUG_MODE = False  

# ==============================
# 📦 CLEARML SETUP
# ==============================
task = Task.init(
    project_name='Mentor Group - Myrthe/Group 1',
    task_name=f'OT2_PPO_Production_{PERSON_NAME}',
    output_uri=True,
    reuse_last_task_id=False # Forces a fresh experiment
)

task.set_base_docker('deanis/2023y2b-rl:latest')
task.set_repo(repo='https://github.com/AndriiRak243703/Y2B25_Task_11.git', branch='hris')
task.set_packages(['stable-baselines3', 'gymnasium', 'pybullet', 'numpy', 'clearml', 'tensorboard'])

# Parse arguments (allows you to tune hyperparameters from ClearML UI later)
parser = argparse.ArgumentParser()
parser.add_argument("--learning_rate", type=float, default=3e-4)
parser.add_argument("--batch_size", type=int, default=256) # Increased for stability
parser.add_argument("--n_steps", type=int, default=2048)
parser.add_argument("--total_timesteps", type=int, default=1_000_000)
parser.add_argument("--gamma", type=float, default=0.99)
args, unknown = parser.parse_known_args()

# Execute remotely if we are NOT debugging
if not DEBUG_MODE:
    task.execute_remotely(queue_name='default')

# ==============================
# 🧪 WRAPPER SETUP
# ==============================
def make_wrapped_env(render=False):
    """
    Creates the OT2 environment. 
    On the server, render is ALWAYS False to save resources.
    """
    env = OT2Env(render=render)
    env = TimeLimit(env, max_episode_steps=1000)
    env = Monitor(env, info_keywords=("is_success", "dist_mm"))
    return env

# ==============================
# 🏁 MAIN TRAINING LOOP
# ==============================
def main():
    print(f"🚀 Training for {PERSON_NAME} | Debug: {DEBUG_MODE}")

    # ⚡ SPEED CONFIGURATION
    # Local Debug: 1 env (to see it visually)
    # Server: 8 envs (to train 8x faster)
    num_envs = 1 if DEBUG_MODE else 8
    
    print(f"Creating {num_envs} parallel environments...")
    
    # We pass 'render=False' explicitly to the vectorized environments
    # This ensures high FPS on the server
    env = make_vec_env(
        lambda: make_wrapped_env(render=DEBUG_MODE), 
        n_envs=num_envs, 
        seed=42
    )

    # Evaluation Environment (Always Headless/No Render)
    eval_env = make_wrapped_env(render=False)
    
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=f"./models/{PERSON_NAME}/",
        log_path=f"./logs/{PERSON_NAME}/",
        eval_freq=max(10000 // num_envs, 1),
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

    print(f"Initialized PPO. Target: {args.total_timesteps} steps.")
    
    model.learn(
        total_timesteps=args.total_timesteps,
        callback=eval_callback,
        progress_bar=True, # Will show progress in ClearML logs too
    )

    # Save final model
    model_name = f"ppo_ot2_final_{PERSON_NAME}.zip"
    model.save(model_name)
    print(f"✅ Training Complete. Saved: {model_name}")
    task.upload_artifact("final_model", artifact_object=model_name)

if __name__ == "__main__":
    main()