import numpy as np
import argparse
from clearml import Task
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from hris_ot2_gym_wrapper import OT2Env

# Custom Callback for real-time monitoring on the ClearML Console
class MaximusCallback(BaseCallback):
    def __init__(self, log_freq=1024):
        super().__init__()
        self.log_freq = log_freq
        print(f"\n{'Step':<10} | {'Error (mm)':<12} | {'Status'}")
        print("-" * 45)

    def _on_step(self) -> bool:
        if self.n_calls % self.log_freq == 0:
            # Extract observation from the local environment variables
            obs = self.locals['new_obs'][0]
            # Calculate distance in mm (assuming normalized coords to mm conversion)
            dist_mm = (np.linalg.norm(obs) / 10.0) * 1000
            
            if dist_mm < 1.0: status = "💎 PERFECT"
            elif dist_mm < 3.0: status = "💰 JACKPOT"
            elif dist_mm < 12.0: status = "⚡ BREAKING"
            else: status = "🔥 HOT"
            
            print(f"{self.num_timesteps:<10} | {dist_mm:>10.2f} mm | {status}")
        return True

def main():
    # --- 1. PROFESSIONAL CONFIGURATION ---
    # Update these to match your branch and filename exactly
    PERSON_NAME = "hris" 
    BRANCH_NAME = "hris/rl-training" 
    ENTRYPOINT = "hris_train_ot2.py"

    # --- 2. HYPERPARAMETER PARSING ---
    # This allows you to change settings from the terminal
    parser = argparse.ArgumentParser()
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--total_timesteps", type=int, default=1000000)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--n_steps", type=int, default=1024)
    args = parser.parse_args()

    # --- 3. CLEARML TASK INITIALIZATION ---
    # Project naming must follow the group standard
    task = Task.init(
        project_name='Mentor Group - Jason/Group 1', 
        task_name='hris_Addiction_Maximus_V2'
    )

    # Set the remote infrastructure configuration
    # Update this line (usually around line 45-50):
    # Change this line in your hris_train_ot2.py script:
    task.set_base_docker('deanis/2023y2b-rl:latest') #
    
    # Trigger remote execution on the GPU queue
    # Ensure this line is exactly:
    task.execute_remotely(queue_name="default") #

    # --- 4. TRAINING DEPLOYMENT ---
    env = OT2Env(render=False)
    
    model = PPO(
        "MlpPolicy", 
        env, 
        learning_rate=args.learning_rate,
        gamma=0.98,           # Value the 1mm jackpot over the 12mm near-miss
        ent_coef=0.02,        # Higher curiosity to break 12mm habits
        n_steps=args.n_steps, 
        batch_size=args.batch_size,
        clip_range=0.1,       # Surgical precision leash
        verbose=1
    )

    print(f"--- DEPLOYING {PERSON_NAME.upper()} ADDICTION MAXIMUS ---")
    
    try:
        model.learn(
            total_timesteps=args.total_timesteps, 
            callback=MaximusCallback()
        )
        # Use scientific notation for the learning rate in the filename for clarity
        model.save(f"models/{PERSON_NAME}/maximus_final_lr{args.learning_rate:.0e}")
    except KeyboardInterrupt:
        model.save(f"models/{PERSON_NAME}/maximus_interrupted")

if __name__ == "__main__":
    main()