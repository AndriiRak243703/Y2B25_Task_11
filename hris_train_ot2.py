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
            if dist_mm < 1.0: status = "💎 PERFECT"
            elif dist_mm < 3.0: status = "💰 JACKPOT"
            elif dist_mm < 12.0: status = "⚡ BREAKING"
            else: status = "🔥 HOT"
            print(f"{self.num_timesteps:<10} | {dist_mm:>10.2f} mm | {status}")
        return True

# ============================================================================
# MAIN
# ============================================================================
def main():
    # 1. NEW TASK NAME (CRITICAL TO FIX THE ERROR)
    # By changing this name, we force ClearML to stop reusing the broken task ID
    PERSON_NAME = "hris"
    UNIQUE_TASK_NAME = f'{PERSON_NAME}_Addiction_Maximus_FRESH_START_V4'

    # 2. Argument Parsing
    parser = argparse.ArgumentParser()
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--total_timesteps", type=int, default=1000000)
    args = parser.parse_args()

    # 3. ClearML Initialization
    task = Task.init(
        project_name='Mentor Group - Jason/Group 1', 
        task_name=UNIQUE_TASK_NAME
    )

    # 4. Explicit Remote Configuration
    # This tells the worker where to get the code
    task.set_repo(
        repo='https://github.com/AndriiRak243703/Y2B25_Task_11.git',
        branch='hris/rl-training'
    )

    # This fixes the "invalid reference format" (No commas allowed!)
    task.set_base_docker('deanis/2023y2b-rl:latest')
    
    # Ensure dependencies are installed on the remote GPU
    task.set_packages(['tensorboard', 'clearml', 'gymnasium', 'stable-baselines3', 'pybullet'])

    # Send to the default queue
    task.execute_remotely(queue_name='default')

    # 5. Training Logic
    env = OT2Env(render=False)
    
    model = PPO(
        "MlpPolicy", 
        env, 
        learning_rate=args.learning_rate,
        gamma=0.98,
        ent_coef=0.02,
        n_steps=1024,
        batch_size=64,
        clip_range=0.1,
        verbose=1
    )

    print(f"--- DEPLOYING {PERSON_NAME.upper()} ADDICTION MAXIMUS ---")
    try:
        model.learn(total_timesteps=args.total_timesteps, callback=MaximusCallback())
        model.save(f"ot2_maximus_final")
    except KeyboardInterrupt:
        model.save(f"ot2_maximus_interrupted")

if __name__ == "__main__":
    main()