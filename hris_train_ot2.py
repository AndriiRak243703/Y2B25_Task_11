import numpy as np
import argparse
from clearml import Task
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from hris_ot2_gym_wrapper import OT2Env

# ============================================================================
# CUSTOM CALLBACK
# ============================================================================
class MaximusCallback(BaseCallback):
    def __init__(self, log_freq=1024):
        super().__init__()
        self.log_freq = log_freq
        print(f"\n{'Step':<10} | {'Error (mm)':<12} | {'Status'}")
        print("-" * 45)

    def _on_step(self) -> bool:
        if self.n_calls % self.log_freq == 0:
            obs = self.locals['new_obs'][0]
            dist_mm = (np.linalg.norm(obs) / 10.0) * 1000
            if dist_mm < 1.0: status = " PERFECT"
            elif dist_mm < 3.0: status = " JACKPOT"
            elif dist_mm < 12.0: status = " BREAKING"
            else: status = "HOT"
            print(f"{self.num_timesteps:<10} | {dist_mm:>10.2f} mm | {status}")
        return True

# ============================================================================
# MAIN
# ============================================================================
def main():
    PERSON_NAME = "hris"
    
    # --- 1. THE CRITICAL REUSE FIX ---
    # reuse_last_task_id=False prevents ClearML from overwriting your broken run
    task = Task.init(
        project_name='Mentor Group - Jason/Group 1', 
        task_name=f'{PERSON_NAME}_Addiction_Maximus_FINAL',
        reuse_last_task_id=False # <--- ADD THIS LINE
    )

    # --- 2. EXPLICIT REMOTE CONFIG ---
    task.set_repo(
        repo='https://github.com/AndriiRak243703/Y2B25_Task_11.git',
        branch='hris/rl-training'
    )

    # This image name MUST be exactly this (no commas!)
    task.set_base_docker('deanis/2023y2b-rl:latest')
    
    task.set_packages(['tensorboard', 'clearml', 'gymnasium', 'stable-baselines3', 'pybullet'])

    # Enqueue to the GPU cluster
    task.execute_remotely(queue_name='default')

    # --- 3. TRAINING LOGIC ---
    env = OT2Env(render=False)
    
    model = PPO(
        "MlpPolicy", 
        env, 
        learning_rate=5e-5,
        gamma=0.98,
        ent_coef=0.02,
        n_steps=1024,
        batch_size=64,
        clip_range=0.1,
        verbose=1
    )

    print(f"--- DEPLOYING {PERSON_NAME.upper()} ADDICTION MAXIMUS ---")
    try:
        model.learn(total_timesteps=2000000, callback=MaximusCallback())
        model.save(f"ot2_maximus_final")
    except KeyboardInterrupt:
        model.save(f"ot2_maximus_interrupted")

if __name__ == "__main__":
    main()