import numpy as np
import argparse
from clearml import Task
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from hris_ot2_gym_wrapper import OT2Env

# Custom Callback for clear terminal output in ClearML Console
class MaximusCallback(BaseCallback):
    def __init__(self, log_freq=1024):
        super().__init__()
        self.log_freq = log_freq
        print(f"\n{'Step':<10} | {'Error (mm)':<12} | {'Status'}")
        print("-" * 45)

    def _on_step(self) -> bool:
        if self.n_calls % self.log_freq == 0:
            obs = self.locals['new_obs'][0]
            # Convert internal coordinates back to mm for the log
            dist_mm = (np.linalg.norm(obs) / 10.0) * 1000
            
            if dist_mm < 1.0: status = "💎 PERFECT"
            elif dist_mm < 3.0: status = "💰 JACKPOT"
            elif dist_mm < 12.0: status = "⚡ BREAKING"
            else: status = "🔥 HOT"
            
            print(f"{self.num_timesteps:<10} | {dist_mm:>10.2f} mm | {status}")
        return True

def main():
    # 1. HYPERPARAMETERS
    parser = argparse.ArgumentParser()
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--total_timesteps", type=int, default=2000000)
    parser.add_argument("--batch_size", type=int, default=64)
    args = parser.parse_args()

    # 2. CLEARML REMOTE CONFIGURATION
    # reuse_last_task_id=False forces a fresh start and clears Docker errors
    task = Task.init(
        project_name='Mentor Group - Jason/Group 1', 
        task_name='hris_Addiction_Maximus_FINAL_DEPLOY',
        reuse_last_task_id=False
    )

    # Tell the remote worker where to find your code
    task.set_repo(
        repo='https://github.com/AndriiRak243703/Y2B25_Task_11.git',
        branch='hris/rl-training'
    )

    # Set the required Docker environment
    task.set_base_docker('deanis/2023y2b-rl:latest')
    
    # Pre-install necessary libraries on the GPU machine
    task.set_packages(['tensorboard', 'clearml', 'gymnasium', 'stable-baselines3', 'pybullet'])

    # Launch on the 'default' compute queue
    task.execute_remotely(queue_name='default')

    # 3. TRAINING DEPLOYMENT
    env = OT2Env(render=False)
    
    model = PPO(
        "MlpPolicy", 
        env, 
        learning_rate=args.learning_rate,
        gamma=0.98,
        ent_coef=0.02,
        n_steps=1024,
        batch_size=args.batch_size,
        clip_range=0.1,
        verbose=1,
        tensorboard_log="./logs/"
    )

    print("--- DEPLOYING HRIS ADDICTION MAXIMUS ---")
    
    try:
        model.learn(
            total_timesteps=args.total_timesteps, 
            callback=MaximusCallback()
        )
        model.save("maximus_model_final")
        task.upload_artifact("trained_model", artifact_object="maximus_model_final.zip")
    except KeyboardInterrupt:
        model.save("maximus_model_interrupted")

if __name__ == "__main__":
    main()