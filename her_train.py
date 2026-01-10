import os
import argparse
from clearml import Task
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from her_wrapper import OT2Env

# ==============================
# 🔧 CONFIGURATION
# ==============================
PERSON_NAME = "241236"
DEBUG_MODE = True  # Keep True to see the single robot and red goal marker

task = Task.init(
    project_name='Mentor Group - Myrthe/Group 1',
    task_name=f'OT2_PPO_SingleEnv_{PERSON_NAME}',
    output_uri=True,
    reuse_last_task_id=False # Forces a new clean experiment
)

# Remote execution logic
if not DEBUG_MODE:
    task.execute_remotely(queue_name='default')

# ==============================
# 🧪 HELPERS
# ==============================
def make_wrapped_env(render=False):
    env = OT2Env(render=render)
    env = Monitor(env, info_keywords=("is_success", "dist_mm"))
    return env

def main():
    # 🏁 Setting n_envs=1 for a clean, non-vibrating visual experience
    num_envs = 1 
    
    # Training Env (Shows the GUI and the Target Marker)
    env = make_vec_env(lambda: make_wrapped_env(render=DEBUG_MODE), n_envs=num_envs)
    
    # Eval Env (Runs in background, no GUI to avoid crashes)
    eval_env = make_wrapped_env(render=False)
    
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=f"./models/{PERSON_NAME}/",
        eval_freq=5000,
        n_eval_episodes=5,
        deterministic=True,
        verbose=1,
    )

    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        batch_size=128,
        n_steps=2048,
        verbose=1,
        tensorboard_log="./ppo_tensorboard/"
    )

    print("🚀 Training starting. You should see ONE robot and a RED sphere target.")
    model.learn(total_timesteps=1_000_000, callback=eval_callback, progress_bar=True)
    
    model.save(f"ppo_ot2_final_{PERSON_NAME}")
    task.upload_artifact("final_model", artifact_object=f"ppo_ot2_final_{PERSON_NAME}.zip")

if __name__ == "__main__":
    main()