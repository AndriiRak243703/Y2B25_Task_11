import numpy as np
import argparse
from clearml import Task
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from hris_ot2_gym_wrapper import OT2Env

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
            if dist_mm < 1.0: status = "💎 PERFECT"
            elif dist_mm < 3.0: status = "💰 JACKPOT"
            elif dist_mm < 12.0: status = "⚡ BREAKING"
            else: status = "🔥 HOT"
            print(f"{self.num_timesteps:<10} | {dist_mm:>10.2f} mm | {status}")
        return True

def main():
    PERSON_NAME = "hris"
    
    # Initialize Task with REUSE=FALSE to clear old Docker errors
    task = Task.init(
        project_name='Mentor Group - Jason/Group 1', 
        task_name=f'{PERSON_NAME}_Addiction_Maximus_FINAL',
        reuse_last_task_id=False
    )

    # Explicitly set the Git repository connection
    task.set_repo(
        repo='https://github.com/AndriiRak243703/Y2B25_Task_11.git',
        branch='hris/rl-training'
    )

    # Set the correct Docker image for the remote worker
    task.set_base_docker('deanis/2023y2b-rl:latest')
    
    # Ensure remote worker installs all logic
    task.set_packages(['tensorboard', 'clearml', 'gymnasium', 'stable-baselines3', 'pybullet'])

    # Enqueue to the remote cluster
    task.execute_remotely(queue_name='default')

    # Environment and Model
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
        verbose=1,
        tensorboard_log=f"runs/{PERSON_NAME}"
    )

    print(f"--- DEPLOYING {PERSON_NAME.upper()} ADDICTION MAXIMUS ---")
    try:
        model.learn(total_timesteps=2000000, callback=MaximusCallback())
        model.save("ot2_maximus_final")
        task.upload_artifact("model", artifact_object="ot2_maximus_final.zip")
    except KeyboardInterrupt:
        model.save("ot2_maximus_interrupted")

if __name__ == "__main__":
    main()